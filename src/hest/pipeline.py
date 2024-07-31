from __future__ import annotations

import os
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import openslide
import yaml
from loguru import logger

from hest.HESTData import XeniumHESTData, load_hest, read_HESTData
from hest.io.seg_readers import read_gdf, write_geojson
from hest.registration import register_dapi_he, warp
from hest.subtyping.atlas import get_atlas_from_name
from hest.subtyping.subtyping import assign_cell_types
from hest.utils import get_name_datetime, verify_paths
from hest.wsi import WSI


def process_xenium(
    config_path: str
):
    """ Proprocess cells and then assign cell types """
    import scanpy as sc
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        
    datetime = get_name_datetime()
    if 'name' in config_dict:   
        name = config_dict['name'] + '_' + datetime
    else:
        name = datetime
        
    exp_dir = config_dict.get('exp_dir', 'results/')
    full_exp_dir = os.path.join(exp_dir, name)
    process_dir = os.path.join(full_exp_dir, 'processed')
    
    gdf = None
    
    if 'data_dir' in config_dict:
        data_dir = config_dict['data_dir']
        raise NotImplementedError()
    elif 'hest_id' in config_dict:
        hest_dir = config_dict.get('hest_dir', 'hest_data')
        verify_paths([hest_dir])
        
        st = load_hest(hest_dir, id_list=[config_dict['hest_id']])[0]
        
    if 'registration' in config_dict:
        reg_dict = config_dict['registration']
        
        warped_cells, warped_nuclei = preprocess_cells_xenium(
            reg_dict['he_path'], 
            reg_dict['dapi_path'],
            reg_dict['cell_bound_path'],
            reg_dict['nucleus_bound_path'],
            reg_dict,
            full_exp_dir
        )

        
    if 'cell_subtyping' in config_dict:
        logger.info('Assigning subtypes to cells...')
        
        subtyping_dict = config_dict['cell_subtyping']
        
        cell_adata = sc.read_10x_h5(subtyping_dict['cell_counts'])
        atlas_name = subtyping_dict['atlas_name']
        method = subtyping_dict['method']
        
        cell_types = assign_cell_types(cell_adata, atlas_name, name='', method=method)
        
        if gdf is not None:
            # TODO merge cell types with gdf
            pass
        
        
    if gdf is not None:
        write_geojson(gdf, os.path.join(process_dir, 'cells.geojson'))
        
        
def preprocess_cells_xenium(
    he_wsi: Union[str, WSI, np.ndarray, openslide.OpenSlide, CuImage],  # type: ignore
    dapi_path: str,
    dapi_cells: gpd.GeoDataFrame,
    dapi_nuclei: gpd.GeoDataFrame,
    reg_config: dict,
    full_exp_dir: str,
    name: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ Find non-rigid transformation from DAPI to H&E and 
    transform dapi_cells and dapi_nuclei to the H&E coordinate system
    """
    
    logger.info('Registering Xenium DAPI to H&E...')
    max_non_rigid_registration_dim_px = reg_config.get('max_non_rigid_registration_dim_px', 10000)
    path_registrar = register_dapi_he(
        he_wsi,
        dapi_path,
        registrar_dir=full_exp_dir,
        name=name,
        max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px
    )
    
    logger.info('Warping shapes to H&E...')
    
    # TODO remove
    #path_registrar = 'results/process_xenium/test_2024_07_23_13_14_17/data/_registrar.pickle'
    
    warped_cells = warp( # TODO need some optimization
        dapi_cells,
        path_registrar=path_registrar,
        curr_slide_name=dapi_path
    )
    
    warped_nuclei = warp( # TODO need some optimization
        dapi_nuclei,
        path_registrar=path_registrar,
        curr_slide_name=dapi_path
    )
    
    return warped_cells, warped_nuclei
    
    
def subtyping_pipeline(
    cell_adata: sc.AnnData,
    gdf: gpd.GeoDataFrame,
    subtyping_conf: dict
) -> sc.AnnData:
    import scanpy as sc
    
    
    atlas_name = subtyping_conf['atlas_name']
    full_atlas = subtyping_conf['full_atlas']
    method = subtyping_conf['method']
    
    matcher_kwargs = subtyping_conf.get('matcher_args', {})
    
    
    adata = assign_cell_types(
        cell_adata, 
        atlas_name, 
        name='BRAC', 
        method=method,
        full_atlas=full_atlas,
        matcher_kwargs=matcher_kwargs
    )
    
    return adata