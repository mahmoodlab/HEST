from __future__ import annotations

import os
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely import Polygon

from hest.io.seg_readers import groupby_shape, read_gdf
from hest.utils import (get_name_datetime,
                        value_error_str, verify_paths)
from hestcore.wsi import WSI


def register_dapi_he(
    he_path: Union[str, WSI, np.ndarray, openslide.OpenSlide, CuImage],  # type: ignore
    dapi_path: str, 
    registrar_dir: str = "results/registration",
    name = None,
    max_non_rigid_registration_dim_px=10000,
    micro_rigid_registrar_cls=None,
    micro_rigid_registrar_params={},
    micro_reg=True,
    check_for_reflections=False,
    reuse_registrar=False
) -> str:
    """ Register the DAPI WSI to HE with a fine-grained ridig + non-rigid transform with Valis

    Args:
        dapi_path (str): path to a dapi WSI
        he_path (str): path to an H&E WSI
        registrar_dir (str, optional): the output base registration directory. Defaults to "results/registration".
        name (str, optional): name of current experiment, the path to the output registrar will be {registrar_dir}/name if name is not None,
            or {registrar_dir}/{date} otherwise. Defaults to None.
        max_non_rigid_registration_dim_px (int, optional): largest edge of both WSI will be downscaled to this dimension during non-rigid registration. Defaults to 10000.
    
    Returns:
        str: path to the resulting Valis registrar
    
    """
    
    try:
        from valis_hest import preprocessing, registration
        from valis_hest.slide_io import BioFormatsSlideReader

        from .SlideReaderAdapter import SlideReaderAdapter
    except Exception:
        import traceback
        traceback.print_exc()
        raise Exception("Valis needs to be installed independently. Please install Valis with `pip install valis-hest`")
        
    #verify_paths([dapi_path, he_path])
    
    if name is None:
        date = get_name_datetime()
        registrar_dir = os.path.join(registrar_dir, date)
    else:
        registrar_dir = os.path.join(registrar_dir, name)

    img_list = [
        he_path,
        dapi_path
    ]
    
    registrar_path = os.path.join(registrar_dir, 'data/_registrar.pickle')

    if reuse_registrar:
        registration.init_jvm()
        return registrar_path
    registrar = registration.Valis(
        '', 
        registrar_dir, 
        reference_img_f=he_path, 
        align_to_reference=True,
        img_list=img_list,
        check_for_reflections=check_for_reflections,
        micro_rigid_registrar_params=micro_rigid_registrar_params,
        micro_rigid_registrar_cls=micro_rigid_registrar_cls
    )

    registrar.register(
        brightfield_processing_cls=preprocessing.HEDeconvolution,
        reader_dict= {
            he_path: [SlideReaderAdapter],
            dapi_path: [BioFormatsSlideReader]
        }
    )

    if micro_reg:
        # Perform micro-registration on higher resolution images, aligning *directly to* the reference image
        registrar.register_micro(
            max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px, 
            align_to_reference=True, 
            brightfield_processing_cls=preprocessing.HEDeconvolution,
            reference_img_f=he_path
        )
    
    return registrar_path
        

def warp_gdf_valis(
    shapes: Union[gpd.GeoDataFrame, str],
    path_registrar: str,
    curr_slide_name: str,
    n_workers=-1
) -> gpd.GeoDataFrame:
    """ Warp some shapes (points or polygons) from an existing Valis registration

    Args:
        shapes (Union[gpd.GeoDataFrame, str]): shapes to warp. A `str` will be interpreted as a path a nucleus shape file, can be .geojson, or xenium .parquet (ex: nucleus_boundaries.parquet)
        path_registrar (str): path to the .pickle file of an existing Valis registrar 

    Returns:
        gpd.GeoDataFrame: warped shapes
    """
    
    try:
        from valis_hest import registration
    except Exception:
        import traceback
        traceback.print_exc()
        raise Exception("Valis needs to be installed independently. Please install Valis with `pip install valis-wsi` or follow instruction on their website")
    
    
    if isinstance(shapes, str):
        gdf = read_gdf(shapes)
    elif isinstance(shapes, gpd.GeoDataFrame):
        gdf = shapes.copy()
    else:
        raise ValueError(value_error_str(shapes, 'shapes'))

    registrar = registration.load_registrar(path_registrar)
    slide_obj = registrar.get_slide(registrar.reference_img_f)
    if isinstance(shapes.iloc[0].geometry, Polygon):
        coords = gdf.geometry.get_coordinates(index_parts=True)
        points_gdf = coords
        idx = coords.index.get_level_values(0)
        points_gdf['_polygons'] = idx # keep track of polygons
        points = list(zip(points_gdf['x'], points_gdf['y']))
    else:
        points_gdf = gdf
        gdf['_polygons'] = np.arange(len(points_gdf))
        points = list(zip(gdf.geometry.x, gdf.geometry.y))
        
    morph = registrar.get_slide(curr_slide_name)
    logger.debug('warp with valis...')
    warped = morph.warp_xy_from_to(points, slide_obj)
    logger.debug('finished warping with valis')
    
    if isinstance(shapes.iloc[0].geometry, Polygon):
        points_gdf['xy'] = list(zip(warped[:, 0], warped[:, 1]))
        aggr_df = groupby_shape(points_gdf, '_polygons', n_threads=0)
        gdf.geometry = aggr_df.geometry
    else:
        gdf.geometry = gpd.points_from_xy(warped[:, 0], warped[:, 1])
    
    return gdf
    

def preprocess_cells_xenium(
    he_wsi: Union[str, WSI, np.ndarray, openslide.OpenSlide, CuImage],  # type: ignore
    dapi_path: str,
    dapi_cells: gpd.GeoDataFrame,
    dapi_nuclei: gpd.GeoDataFrame,
    dapi_transcripts: pd.DataFrame,
    reg_config: dict,
    full_exp_dir: str,
    registration_kwargs = {}
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ Find non-rigid transformation from DAPI to H&E and 
    transform dapi_cells, dapi_nuclei and transcripts to the H&E coordinate system
    
    returns (warped_cells, warped_nuclei)
    """

    logger.info('Registering Xenium DAPI to H&E...')
    max_non_rigid_registration_dim_px = reg_config.get('max_non_rigid_registration_dim_px', 10000)
    path_registrar = register_dapi_he(
        he_wsi,
        dapi_path,
        registrar_dir=full_exp_dir,
        name='registration',
        max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px,
        **registration_kwargs
    )
    
    if dapi_transcripts:
        logger.info('Warping transcripts from DAPI to H&E...')
        transcripts_gdf = gpd.GeoDataFrame(dapi_transcripts, geometry=gpd.points_from_xy(dapi_transcripts['dapi_x'], dapi_transcripts['dapi_y']))
        warped_transcripts = warp_gdf_valis( # TODO valis interpolation is slow
            transcripts_gdf,
            path_registrar=path_registrar,
            curr_slide_name=dapi_path
        )
        warped_transcripts = warped_transcripts.drop(['_polygons'], axis=1)
        warped_transcripts['he_x'] = warped_transcripts.geometry.x
        warped_transcripts['he_y'] = warped_transcripts.geometry.y
    else:
        warped_transcripts = None

    if dapi_cells is not None:
        logger.info('Warping cells from DAPI to H&E...')
        warped_cells = warp_gdf_valis( # TODO valis interpolation is slow
            dapi_cells,
            path_registrar=path_registrar,
            curr_slide_name=dapi_path
        )
    else:
        warped_cells = None
    
    if dapi_nuclei is not None:
        logger.info('Warping nuclei from DAPI to H&E...')
        warped_nuclei = warp_gdf_valis( # TODO valis interpolation is slow
            dapi_nuclei,
            path_registrar=path_registrar,
            curr_slide_name=dapi_path
        )
    else:
        warped_nuclei = None
    
    return warped_cells, warped_nuclei, warped_transcripts