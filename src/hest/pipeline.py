import os

import yaml
from loguru import logger

from hest.HESTData import XeniumHESTData, load_hest, read_HESTData
from hest.io.seg_readers import read_gdf, write_geojson
from hest.readers import XeniumReader
from hest.registration import register_dapi_he, warp
from hest.subtyping.atlas import get_atlas_from_name
from hest.subtyping.subtyping import assign_cell_types
from hest.utils import get_name_datetime


def process_xenium(
    config_path: str
):
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
    
    data_dir = config_dict['data_dir']
    st = load_hest('hest_data', id_list=['TENX95'])[0]
    gdf = st.shapes
    
    
    if 'registration' in config_dict:
        reg_dict = config_dict['registration']
        
        logger.info('Registering Xenium DAPI to H&E...')
        max_non_rigid_registration_dim_px = reg_dict.get('max_non_rigid_registration_dim_px', 10000)
        #path_registrar = register_dapi_he(
        #    reg_dict['dapi_path'],
        #    reg_dict['he_path'],
        #    registrar_dir=full_exp_dir,
        #    name='registration',
        #    max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px)
        
        logger.info('Warping shapes to H&E...')
        
        path_registrar = 'results/process_xenium/test_2024_07_23_13_14_17/data/_registrar.pickle'
        
        gdf = warp(
            gdf,
            path_registrar=path_registrar
        )
        
    elif 'affine' in config_dict:
        #TODO
        pass

        
    if 'cell_subtyping' in config_dict:
        logger.info('Assigning subtypes to cells...')
        
        subtyping_dict = config_dict['cell_subtyping']
        
        cell_adata = sc.read_10x_h5ad(subtyping_dict['cell_counts'])
        atlas_name = subtyping_dict['atlas_name']
        method = subtyping_dict['method']
        
        cell_types = assign_cell_types(cell_adata, atlas_name, name='', method=method)
        
        if gdf is not None:
            # TODO merge cell types with gdf
            pass
        
        
    if gdf is not None:
        write_geojson(gdf, os.path.join(process_dir, 'cells.geojson'))