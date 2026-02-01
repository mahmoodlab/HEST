from __future__ import annotations

import gc
import os
from typing import Optional, Tuple, Union
import warnings

import geopandas as gpd
import numpy as np
from loguru import logger

from hest.io.seg_readers import HESTXeniumTranscriptsReader, XeniumTranscriptsReader, groupby_shape, read_gdf, write_geojson
from hest.utils import (deprecated, get_name_datetime, merge_parquet,
                        value_error_str)
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
    except Exception as e:
        raise Exception("Valis needs to be installed independently. Please install Valis with `pip install valis-hest`") from e
        
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

    registration.init_jvm()
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
        
        
def _warp_gdf_valis(gdf, registrar, curr_slide_name, slide_obj):
    if len(gdf) == 0:
        return gdf
    
    geom_type = gdf.geometry.iloc[0].geom_type
    
    if geom_type in ['Polygon', 'MultiPolygon']:
        coords = gdf.geometry.get_coordinates(index_parts=True)
        points_gdf = coords
        idx = coords.index.get_level_values(0)
        points_gdf['_polygons'] = idx 
        points = list(zip(points_gdf['x'], points_gdf['y']))
    elif geom_type == 'Point':
        points_gdf = gdf
        points = list(zip(gdf.geometry.x, gdf.geometry.y))
    else:
        raise NotImplementedError('')
        
    morph = registrar.get_slide(curr_slide_name)
    warped = morph.warp_xy_from_to(points, slide_obj)
    
    if geom_type in ['Polygon', 'MultiPolygon']:
        points_gdf['xy'] = list(zip(warped[:, 0], warped[:, 1]))
        aggr_df = groupby_shape(points_gdf, '_polygons', n_threads=0)
        gdf.geometry = aggr_df.geometry
    else:
        import geopandas as gpd
        gdf.geometry = gpd.points_from_xy(warped[:, 0], warped[:, 1])
        
        
    return gdf

def warp_gdf_valis(
    shapes: Union[gpd.GeoDataFrame, str, dgpd.GeoDataFrame],
    path_registrar: str,
    curr_slide_name: str,
    n_workers=-1,
    use_dask=True
) -> Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]:
    """ Warp some shapes (points or polygons) from an existing Valis registration registrar

    Args:
        shapes (Union[gpd.GeoDataFrame, str, dgpd.GeoDataFrame]): shapes to warp. A `str` will be interpreted as a path a nucleus shape file, can be .geojson, or xenium .parquet (ex: nucleus_boundaries.parquet)
        path_registrar (str): path to the .pickle file of an existing Valis registrar 
        curr_slide_name (str): dapi slide filename in the Valis registrar
        n_workers (int, optional): **Deprecated**. Use dask instead
        use_dask (bool, optional): whenever to use dask to process larger than RAM data, highly recommended for all Xenium samples. Defaults to True.

    Returns:
        Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]: warped geodataframe such that warped shapes are in the geometry column.
    """
    
    if n_workers != -1:
        warnings.warn(
            "The 'n_workers' parameter is deprecated and will be removed in a future version. "
            "Please use 'use_dask' instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
    try:
        from valis_hest import registration
    except Exception:
        import traceback
        traceback.print_exc()
        raise Exception("Valis needs to be installed independently. Please install Valis with `pip install valis-wsi` or follow instruction on their website")
    
    XENIUM_PIXEL_SIZE_MORPH = 0.2125
    
    if isinstance(shapes, str):
        gdf = read_gdf(shapes, reader_kwargs={'pixel_size_morph': XENIUM_PIXEL_SIZE_MORPH, 'use_dask': use_dask})
    elif isinstance(shapes, gpd.GeoDataFrame):
        gdf = shapes.copy()
    else:
        try:
            import dask_geopandas
        except:
            pass
        if dask_geopandas and isinstance(shapes, dask_geopandas.expr.GeoDataFrame):
            gdf = shapes
        else:
            raise ValueError(value_error_str(shapes, 'shapes'))

    from valis_hest.registration import init_jvm
    init_jvm(mem_gb=1)
    registrar = registration.load_registrar(path_registrar)
    slide_obj = registrar.get_slide(registrar.reference_img_f)
    if use_dask:
        from geopandas.array import GeometryDtype
        from distributed import get_client
        
        try:
            client = get_client()
        except:
            client = None
        
        meta = gdf.head(0).copy()

        from geopandas.array import GeometryDtype
        meta['geometry'] = gpd.GeoSeries(dtype=GeometryDtype())

        meta = meta.set_geometry('geometry').set_crs("EPSG:4326")
        if client is None:
            registrar_future = registrar
            slide_obj_future = slide_obj
        else:
            [registrar_future] = client.scatter([registrar], broadcast=True)
            [slide_obj_future] = client.scatter([slide_obj], broadcast=True)
        gdf = gdf.map_partitions(_warp_gdf_valis, registrar_future, curr_slide_name, slide_obj_future, meta=meta)
    else:
        gdf = _warp_gdf_valis(gdf, registrar, curr_slide_name, slide_obj)
    
    return gdf


def _read_transcripts(
    dapi_transcripts: str,
    use_dask: bool,
):
    PIXEL_SIZE_MORPH = 0.2125
    
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(dapi_transcripts)
    if 'dapi_x' in parquet_file.schema.names:
        reader = HESTXeniumTranscriptsReader(use_dask=use_dask)
    else:
        reader = XeniumTranscriptsReader(pixel_size_morph=PIXEL_SIZE_MORPH, use_dask=use_dask)
    
    gdf_transcripts = reader.read_gdf(dapi_transcripts)
    return gdf_transcripts


def _format_transcripts(warped_transcripts):
    if '_polygons' in warped_transcripts.columns:
        warped_transcripts = warped_transcripts.drop(['_polygons'], axis=1)
    warped_transcripts['he_x'] = warped_transcripts.geometry.x
    warped_transcripts['he_y'] = warped_transcripts.geometry.y
    return warped_transcripts


def _warp_transcripts(
    dapi_transcripts,
    use_dask,
    verbose,
    path_registrar,
    dapi_path
):
    gdf_transcripts = _read_transcripts(dapi_transcripts, use_dask)
    
    if verbose:
        logger.info('Warping transcripts from DAPI to H&E...')
        
    warped_transcripts = warp_gdf_valis(
        gdf_transcripts,
        path_registrar=path_registrar,
        curr_slide_name=dapi_path,
        use_dask=use_dask,
    )
    
    warped_transcripts = _format_transcripts(warped_transcripts)
    return warped_transcripts


def warp_xenium_objects(
    path_registrar: str, 
    dapi_path: str,
    dapi_cells: str=None,
    dapi_transcripts: str=None,
    dapi_nuclei: str=None,
    use_dask=True,
    verbose=True
) -> Tuple[Optional[Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]], 
                 Optional[Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]], 
                 Optional[Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]]]:
    """ **Deprecated** Use warp_and_save_xenium_objects instead. Wrap Xenium transcripts, cells and nuclei using Valis non-rigid micro-registration. 

    Args:
        path_registrar (str): path to an existing Valis registrar
        dapi_path (str): dapi slide filename in the Valis registrar
        dapi_cells (str, optional): path to xenium .parquet cell bondaries, usually **/cell_boundaries.parquet. Defaults to None.
        dapi_transcripts (str, optional): path to xenium .parquet nucleus bondaries, usually **/nucleus_boundaries.parquet. Defaults to None.
        dapi_nuclei (str, optional): path to xenium .parquet transcripts, usually **/transcripts.parquet. Defaults to None.
        use_dask (bool, optional): whenever to use dask to process larger than RAM data, highly recommended for all Xenium samples. Defaults to True.
        verbose (bool, optional): verbose flag. Defaults to True.

    Returns:
        warped (Tuple): warped objects as (warped_cells, warped_nuclei, warped_transcripts)
        
    Example:
            >>> registrar_path = "./valis_results/data/_registrar.pickle"
            >>> cells_path = "./xenium_out/cell_boundaries.parquet"
            >>> cells, nuclei, transcripts = warp_xenium_objects(
            ...     path_registrar=registrar_path,
            ...     dapi_path="morphology_focus.ome.tif",
            ...     dapi_cells=cells_path,
            ...     use_dask=True
            ... )
            >>> if cells is not None:
            ...     print(f"Warped {len(cells)} cells using Dask: {type(cells)}")
    """
    warnings.warn(
        "warp_xenium_objects is deprecated and will be removed in a future version. "
        "Please use 'warp_and_save_xenium_objects' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if dapi_transcripts:
        warped_transcripts = _warp_transcripts(dapi_transcripts, use_dask, 
                                               verbose, path_registrar, dapi_path)
    else:
        warped_transcripts = None

    if dapi_cells is not None:
        
        if verbose:
            logger.info('Warping cells from DAPI to H&E...')
        
        warped_cells = warp_gdf_valis(
            dapi_cells,
            path_registrar=path_registrar,
            curr_slide_name=dapi_path,
            use_dask=use_dask,
        )
    else:
        warped_cells = None
    
    if dapi_nuclei is not None:
        
        if verbose:
            logger.info('Warping nuclei from DAPI to H&E...')
        
        warped_nuclei = warp_gdf_valis(
            dapi_nuclei,
            path_registrar=path_registrar,
            curr_slide_name=dapi_path,
            use_dask=use_dask,
        )
    else:
        warped_nuclei = None

    return warped_cells, warped_nuclei, warped_transcripts


def warp_and_save_xenium_objects(
    path_registrar: str, 
    dapi_path: str,
    save_dir: str,
    dapi_cells: str=None,
    dapi_transcripts: str=None,
    dapi_nuclei: str=None,
    use_dask=True,
    verbose=True,
    save_parquet=True,
    save_geojson=True,
) -> None:
    """ Wrap Xenium transcripts, cells and nuclei using Valis non-rigid micro-registration and save them.

    Args:
        path_registrar (str): path to an existing Valis registrar
        dapi_path (str): dapi slide filename in the Valis registrar
        save_dir (str): where to save warped objects. Objects will be saved to:
            - save_dir/he_cell_seg.parquet
            - save_dir/he_nucleus_seg.parquet
            - save_dir/aligned_transcripts
        dapi_cells (str, optional): path to xenium .parquet cell bondaries, usually **/cell_boundaries.parquet. Defaults to None.
        dapi_transcripts (str, optional): path to xenium .parquet nucleus bondaries, usually **/nucleus_boundaries.parquet. Defaults to None.
        dapi_nuclei (str, optional): path to xenium .parquet transcripts, usually **/transcripts.parquet. Defaults to None.
        use_dask (bool, optional): whenever to use dask to process larger than RAM data, highly recommended for all Xenium samples. Defaults to True.
        verbose (bool, optional): verbose flag. Defaults to True.
        save_parquet (bool, optional): whenever to save objects as parquet. Defaults to True.
        save_geojson (bool, optional): whenever to save objects as geojson. Defaults to True.

    Example:
            >>> registrar_path = "./valis_results/data/_registrar.pickle"
            >>> cells_path = "./xenium_out/cell_boundaries.parquet"
            >>> warp_and_save_xenium_objects(
            ...     path_registrar=registrar_path,
            ...     save_dir="warped_xenium",
            ...     dapi_path="morphology_focus.ome.tif",
            ...     dapi_cells=cells_path,
            ...     use_dask=True
            ... )
            >>> if cells is not None:
            ...     print(f"Warped {len(cells)} cells using Dask: {type(cells)}")
    """
    if not os.path.exists(save_dir):
        raise ValueError(f"Save directory '{save_dir}' doesn't exist.")
    
    if dapi_transcripts:
        warped_transcripts = _warp_transcripts(dapi_transcripts, use_dask, 
                                               verbose, path_registrar, dapi_path)

        res_path = 'aligned_transcripts' if use_dask else 'aligned_transcripts.parquet'
        warped_transcripts.to_parquet(os.path.join(save_dir, res_path))
        
        if use_dask:
            merge_parquet(os.path.join(save_dir, 'aligned_transcripts'),
                        os.path.join(save_dir, 'aligned_transcripts.parquet'))

        del warped_transcripts
        gc.collect()
    else:
        warped_transcripts = None
        
    
    if dapi_cells is not None:
        
        if verbose:
            logger.info('Warping cells from DAPI to H&E...')
        
        warped_cells = warp_gdf_valis(
            dapi_cells,
            path_registrar=path_registrar,
            curr_slide_name=dapi_path,
            use_dask=use_dask,
        )
        
        if save_parquet:
            res_path = 'he_cell_seg' if use_dask else 'he_cell_seg.parquet'
            warped_cells.to_parquet(os.path.join(save_dir, res_path))
            
            if use_dask:
                merge_parquet(os.path.join(save_dir, 'he_cell_seg'),
                            os.path.join(save_dir, 'he_cell_seg.parquet'))

        if save_geojson:
            write_geojson(warped_cells.compute(), os.path.join(save_dir, f'he_cell_seg.geojson'))
            

        del warped_cells
        gc.collect()
    else:
        warped_cells = None
        
    
    if dapi_nuclei is not None:
        
        if verbose:
            logger.info('Warping nuclei from DAPI to H&E...')
        
        warped_nuclei = warp_gdf_valis(
            dapi_nuclei,
            path_registrar=path_registrar,
            curr_slide_name=dapi_path,
            use_dask=use_dask,
        )
        
        if save_parquet:
            res_path = 'he_nucleus_seg' if use_dask else 'he_nucleus_seg.parquet'
            warped_nuclei.to_parquet(os.path.join(save_dir, res_path))
            
            if use_dask:
                merge_parquet(os.path.join(save_dir, 'he_nucleus_seg'),
                            os.path.join(save_dir, 'he_nucleus_seg.parquet'))
            
        if save_geojson:
            write_geojson(warped_nuclei.compute(), os.path.join(save_dir, f'he_nucleus_seg.geojson'))
            
        del warped_nuclei
        gc.collect()
    else:
        warped_nuclei = None
    
@deprecated
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

    return warp_xenium_objects(
        path_registrar,
        dapi_path,
        dapi_cells,
        dapi_transcripts,
        dapi_nuclei
    )