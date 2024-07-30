import os
import tempfile
from typing import Union

import geopandas as gpd
import numpy as np
from shapely import Polygon
from tqdm import tqdm

from hest.io.seg_readers import read_gdf
from hest.utils import get_name_datetime, value_error_str


def register_dapi_he(
    dapi_path: str, 
    he_path: str, 
    registrar_dir: str = "results/registration",
    name = None,
    max_non_rigid_registration_dim_px=10000
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
        from valis import preprocessing, registration
        from valis.slide_io import BioFormatsSlideReader
    except Exception:
        import traceback
        traceback.print_exc()
        raise Exception("Valis needs to be installed independently. Please install Valis with `pip install valis-wsi` or follow instruction on their website")
        
    
    verify_paths_exist(paths=[dapi_path, he_path])
    
    if name is None:
        date = get_name_datetime()
        registrar_dir = os.path.join(registrar_dir, date)
    else:
        registrar_dir = os.path.join(registrar_dir, name)
        

    img_list = [
        dapi_path,
        he_path
    ]

    registrar = registration.Valis(
        '', 
        registrar_dir, 
        reference_img_f=he_path, 
        align_to_reference=True,
        img_list=img_list)

    registrar.register(
        brightfield_processing_cls=preprocessing.HEDeconvolution,
        reader_cls=BioFormatsSlideReader
    )

    # Perform micro-registration on higher resolution images, aligning *directly to* the reference image
    registrar.register_micro(
        max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px, 
        align_to_reference=True, 
        brightfield_processing_cls=preprocessing.HEDeconvolution,
        reference_img_f=he_path
    )
    
    return os.path.join(registrar_dir, 'data/_registrar.pickle')
        

def warp(
    shapes: Union[gpd.GeoDataFrame, str],
    path_registrar: str
) -> gpd.GeoDataFrame:
    """ Warp some shapes (points or polygons) from an existing Valis registration

    Args:
        shapes (Union[gpd.GeoDataFrame, str]): shapes to warp. A `str` will be interpreted as a path a nucleus shape file, can be .geojson, or xenium .parquet (ex: nucleus_boundaries.parquet)
        path_registrar (str): path to the .pickle file of an existing Valis registrar 

    Returns:
        gpd.GeoDataFrame: warped shapes
    """
    
    try:
        from valis import registration
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
    
    
    registration.init_jvm()

    registrar = registration.load_registrar(path_registrar)
    slide_obj = registrar.get_slide(registrar.reference_img_f)

    if isinstance(shapes.iloc[0].geometry, Polygon):
        gdf['points'] = [list(polygon.exterior.coords) for polygon in gdf.geometry]
        gdf['_cell_id'] = np.arange(len(gdf))
        point_gdf = gdf.explode('points')
        point_gdf['points'] = point_gdf['points'].apply(lambda x: list(x))
        points = list(point_gdf['points'].values)
    else:
        point_gdf = gdf
        gdf['_cell_id'] = np.arange(len(point_gdf))
        points = list(zip(gdf.geometry.x, gdf.geometry.y))
        
    warped = slide_obj.warp_xy(points)
    point_gdf['warped'] = warped.tolist()

    aggr_df = point_gdf.groupby('_cell_id').agg({
        'warped': list
        }
    )
    
    polygons = [Polygon(x) for x in aggr_df['warped']]
    gdf = gdf.drop(['_cell_id'], axis=1)
    
    gdf.geometry = polygons
    
    registration.kill_jvm()
    return gdf
    