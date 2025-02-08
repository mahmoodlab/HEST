from __future__ import annotations

import gc
import json
import os
import traceback
from abc import abstractmethod
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import openslide
import pandas as pd
import yaml
from hestcore.wsi import WSI
from loguru import logger
from tqdm import tqdm

from hest.HESTData import (VisiumHDHESTData, XeniumHESTData)
from hest.io.seg_readers import GeojsonCellReader, read_gdf, write_geojson
from hest.readers import VisiumHDReader, XeniumReader, read_and_save
from hest.registration import preprocess_cells_xenium, register_dapi_he, warp_gdf_valis
from hest.segmentation.cell_segmenters import (bin_per_cell,
                                               cell_segmenter_factory)
from hest.subtyping.subtyping import assign_cell_types
from hest.utils import (ALIGNED_HE_FILENAME, check_arg,
                        find_first_file_endswith, get_col_selection,
                        get_path_from_meta_row,
                        print_resource_usage, visualize_random_crops)


class ProcessingPipeline:
    st = None
    
    def __init__(self, config, full_exp_dir):
        self.config = config
        self.full_exp_dir = full_exp_dir
    
    @abstractmethod
    def on_skip_preprocessing(self, nuc_gdf, cell_adata) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
        pass
    
    @abstractmethod
    def preprocess(self) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        pass
    
    
class XeniumProcessingPipeline(ProcessingPipeline):
    
    def on_skip_preprocessing(self, nuc_gdf, cell_adata) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
        if nuc_gdf is None:
            pass #TODO
        if cell_adata is None:
            import scanpy as sc
            data_dir = self.config.get('data_dir')
            st = XeniumReader().auto_read(data_dir, load_img=False)
            cell_adata = sc.read_h5ad(st.cell_adata_path)
        return cell_adata, nuc_gdf
    
    def preprocess(self) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        preprocessing_conf = self.config.get('preprocessing', None)
        data_dir = self.config.get('data_dir')
        
        st = XeniumReader().auto_read(data_dir)
        
        reg_config = preprocessing_conf.get('registration', {})
        
        cell_adata = st.cell_adata
        
        cell_gdf, nuc_gdf  = preprocess_cells_xenium(
            st.wsi,
            st.dapi_path,
            st.get_shapes('tenx_cells', 'dapi').shapes,
            st.get_shapes('tenx_nuclei', 'dapi').shapes,
            reg_config,
            self.full_exp_dir
        )
        
        return cell_adata, cell_gdf, nuc_gdf


class VisiumHDProcessingPipeline(ProcessingPipeline):
    
    def on_skip_preprocessing(self, nuc_gdf, cell_adata) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
        if cell_adata is None:
            raise ValueError('cell_adata_path is required if preprocessing is skipped')
        return cell_adata, nuc_gdf
    
    def preprocess(self) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        data_dir = self.config.get('data_dir')
        st = VisiumHDReader().auto_read(data_dir)
        
        bc_matrix_2um_path = find_first_file_endswith(st.square_2um_path, 'filtered_feature_bc_matrix.h5')
        bin_positions_2um_path = find_first_file_endswith(st.square_2um_path, 'tissue_positions.parquet')
        
        if bc_matrix_2um_path is None or bin_positions_2um_path is None:
            raise FileNotFoundError(f"Make sure that your directory {data_dir} has a square_002um folder contaning those files: filtered_feature_bc_matrix.h5, tissue_positions.parquet")
        
        preprocessing_conf = self.config.get('preprocessing', None)
        
        segment_config = preprocessing_conf.get('segmentation', {})
        binning_config = preprocessing_conf.get('cell_binning', {})
        
        nuclei_path = preprocessing_conf.get('nuclei_path', None)
        if nuclei_path is not None:
            logger.info("nuclei_path is specified, bypass segmentation")
        
        cell_adata, cell_gdf, nuc_gdf = preprocess_cells_visium_hd(
            st.wsi,
            self.full_exp_dir,
            st.pixel_size,
            bc_matrix_2um_path,
            bin_positions_2um_path,
            segment_config,
            binning_config,
            segment_config.get('method', 'cellvit'),
            nuclei_path
        )
        
        return cell_adata, cell_gdf, nuc_gdf
    

def process_from_config(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    if 'preprocessing' not in config and 'cell_subtyping' not in config:
        raise ValueError("Please provide at least one of these steps in config: ['preprocessing', 'cell_subtyping']")
        
    result_dir = config.get('results_dir')
    name = config.get('name', '')
    full_exp_dir = os.path.join(result_dir, name)
    os.makedirs(full_exp_dir, exist_ok=True)
    
    technology = config.get('technology').lower()
    check_arg(technology, 'technology', ['xenium', 'visium-hd'])
    
    preprocessing_conf = config.get('preprocessing', None)
    
    if technology == 'xenium':
        pipeline = XeniumProcessingPipeline(config, full_exp_dir)
    elif technology == 'visium-hd':
        pipeline = VisiumHDProcessingPipeline(config, full_exp_dir)
    
    # Can bypass preprocessing
    if preprocessing_conf is None:
        logger.info("no 'preprocessing' key found in config, skip preprocessing")
        
        # Try to infer cell_adata, nuc_gdf if preprocessing is skipped
        nuclei_path = config.get('nuclei_path', None)
        cell_adata_path = config.get('cell_adata_path', None)
        
        if nuclei_path is not None:
            nuc_gdf = read_gdf(nuclei_path)
        else:
            nuc_gdf = None
        if cell_adata_path is not None:
            import scanpy as sc
            cell_adata = sc.read_10x_h5(cell_adata_path) # TODO handle non 10x with catch
        else:
            cell_adata = None
        
        if nuclei_path is None or cell_adata_path is None:
            logger.warning(f"No 'nuclei_path' or 'cell_adata_path' detected")
        cell_adata, nuc_gdf = pipeline.on_skip_preprocessing(nuc_gdf, cell_adata)
    else:
        cell_adata, _, nuc_gdf = pipeline.preprocess()
        
        
    if nuc_gdf is None:
        logger.warning("No nuclear segmentation detected, make sure to enable preprocessing or to pass a custom 'nuclei_path'")
            
    subtyping_conf = config.get('cell_subtyping', None)
        
    if subtyping_conf is None:
        logger.info("no 'cell_subtyping' key found in config, skip subtyping")
    else:
        matcher_kwargs = subtyping_conf.pop('matcher', {})
        atlas_name = matcher_kwargs.pop('atlas_name', None)
        
        types_adata, gdf_types = subtyping_pipeline(cell_adata, atlas_name, full_exp_dir, nuc_gdf, matcher_kwargs=matcher_kwargs, **subtyping_conf)
                
        types_adata.write_h5ad(os.path.join(full_exp_dir, 'types_adata.h5ad'))
        
    
    
def subtyping_pipeline(
    cell_adata: sc.AnnData,
    atlas_name: str,
    full_exp_dir,
    gdf: gpd.GeoDataFrame=None,
    matcher_kwargs: dict={},
    save_geojson=True,
    save_parquet=True,
    subtypes_path=None
) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
    import scanpy as sc

    # full_atlas = subtyping_conf.get('full_atlas')
    # method = subtyping_conf['method']

    # matcher_kwargs = subtyping_conf.get('matcher_args', {})
    if 'cell_id' not in gdf.columns:
        raise ValueError("gdf needs to contain a 'cell_id' column")
    
    if subtypes_path is None:
        adata = assign_cell_types(
            cell_adata, 
            atlas_name,
            '',
            **matcher_kwargs
        )
    else:
        subtypes = pd.read_csv(subtypes_path, index_col=0)
        cell_adata.obs['cell_type_pred'] = subtypes['Cluster']
        na_mask = cell_adata.obs['cell_type_pred'].isna()
        nb_na = na_mask.sum()
        if nb_na > 0:
            logger.warning(f"{nb_na} unattributed cells in ground truth file {subtypes_path}. Mark unmatched cells as 'Unknown'")
        cell_adata.obs.loc[na_mask, 'cell_type_pred'] = 'Unknown'
        adata = cell_adata
    
    if gdf is not None:
        gdf['cell_id'] = gdf['cell_id'].astype(str)
        gdf['cell_type_pred'] = adata.obs.loc[gdf['cell_id'], 'cell_type_pred'].values
        
    if gdf is not None:
        if save_geojson:
            write_geojson(gdf, os.path.join(full_exp_dir, 'nuclei_types.geojson'), 'cell_type_pred')
        if save_parquet:
            gdf.to_parquet('nuclei_types.parquet')
    
    return adata, gdf


def preprocess_cells_visium_hd(
    he_wsi: Union[str, WSI, np.ndarray, openslide.OpenSlide, CuImage],  # type: ignore
    full_exp_dir: str,
    pixel_size: str,
    bc_matrix_2um_path: str,
    bin_positions_2um_path: str,
    segment_kwargs: dict = {},
    binning_kwargs: dict = {},
    segment_method = 'cellvit',
    nuclei_path = None
) -> Tuple[sc.AnnData, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    

    
    if nuclei_path is None:
        segmenter = cell_segmenter_factory(segment_method)
        logger.info('Segmenting cells...')
        path_geojson = segmenter.segment_cells(he_wsi, 'seg', pixel_size, save_dir=full_exp_dir, **segment_kwargs)
        nuc_gdf = GeojsonCellReader().read_gdf(path_geojson)  
    else:
        nuc_gdf = read_gdf(nuclei_path)
    
    
    logger.info('Expanding nuclei/binning expression per cell...')
    cell_adata, cell_gdf = bin_per_cell(
        nuc_gdf, 
        bc_matrix_2um_path,
        bin_positions_2um_path,
        pixel_size=pixel_size
    )
    
    cell_adata.write_h5ad(os.path.join(full_exp_dir, f'cell_bin.h5'))
    
    return cell_adata, cell_gdf, nuc_gdf


def process_meta_df(
    meta_df, 
    save_spatial_plots=True, 
    pyramidal=True, 
    save_img=True, 
    preprocess=False, 
    no_except=False, 
    segment_tissue=True,
    registration_kwargs={},
    read_kwargs={},
    segment_kwargs={},
    preprocess_kwargs={}
):
    """Internal use method, process all the raw ST data in the meta_df"""
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        try:
            print_resource_usage()
            
            path = get_path_from_meta_row(row)
            bigtiff = not(isinstance(row['bigtiff'], float) or row['bigtiff'] == 'FALSE')
            save_kwargs = {'save_cell_seg': True, 'save_nuclei_seg': True, 'save_transcripts': True} if row['st_technology'].lower() == 'xenium' and not preprocess else {}
            st = read_and_save(
                path, 
                save_plots=save_spatial_plots, 
                pyramidal=pyramidal, 
                bigtiff=bigtiff, 
                plot_pxl_size=True, 
                save_img=save_img, 
                segment_tissue=segment_tissue, 
                read_kwargs=read_kwargs,
                save_kwargs=save_kwargs,
                segment_kwargs=segment_kwargs
            )

            # TODO register segmentation for xenium and save
            if preprocess:
                
                full_exp_dir = os.path.join('results', 'preprocessing', row['id'])
                if isinstance(st, XeniumHESTData):
                    
                    print('read shapes')
                    for shape in st.shapes:
                        
                        if shape.name == 'tenx_cell' and shape.coordinate_system == 'dapi':
                            dapi_cells = shape.shapes
                        elif shape.name == 'tenx_nucleus' and shape.coordinate_system == 'dapi':
                            dapi_nuclei = shape.shapes
                        
                    print('finished reading shapes')
                    reg_config = {}
                    
                    alignment_file_path = st.alignment_file_path if registration_kwargs.get('affine', False) else None
                        
                    warped_cells, warped_nuclei, st.transcript_df = preprocess_cells_xenium(
                        os.path.join(path, 'processed', ALIGNED_HE_FILENAME), 
                        st.dapi_path,
                        dapi_cells,
                        dapi_nuclei,
                        st.transcript_df,
                        reg_config,
                        full_exp_dir,
                        registration_kwargs=registration_kwargs,
                        alignment_file_path=alignment_file_path
                    )

                    
                    print('Saving warped cells/nuclei...')
                    warped_cells.to_parquet(os.path.join(path, 'processed', f'he_cell_seg.parquet'))
                    warped_nuclei.to_parquet(os.path.join(path, 'processed', f'he_nucleus_seg.parquet'))
                    st.transcript_df.to_parquet(os.path.join(path, 'processed', f'aligned_transcripts.parquet'))
                    write_geojson(warped_cells, os.path.join(path, 'processed', f'he_cell_seg.geojson'), '', chunk=True)
                    write_geojson(warped_nuclei, os.path.join(path, 'processed', f'he_nucleus_seg.geojson'), '', chunk=True)
                elif isinstance(st, VisiumHDHESTData):
                    segment_config = {}
                    binning_config = {}
                    
                    bc_matrix_path = find_first_file_endswith(os.path.join(path, 'binned_outputs', 'square_002um'), 'filtered_feature_bc_matrix.h5')
                    bin_positions_path = find_first_file_endswith(os.path.join(path, 'binned_outputs', 'square_002um', 'spatial'), 'tissue_positions.parquet')
                    
                    del st.wsi
                    preprocess_cells_visium_hd(
                        os.path.join(path, 'processed', ALIGNED_HE_FILENAME),
                        full_exp_dir,
                        st.pixel_size,
                        bc_matrix_path,
                        bin_positions_path,
                        segment_config,
                        binning_config,
                        **preprocess_kwargs
                    )

            if isinstance(st, XeniumHESTData):
                visualize_random_crops(st.transcript_df, st.wsi, './', st.get_shapes('tenx_nucleus', 'he').shapes)
            
            row_dict = row.to_dict()

            
            # remove all whitespace values
            row_dict = {k: (np.nan if isinstance(v, str) and not v.strip() else v) for k, v in row_dict.items()}
            combined_meta = {**st.meta, **row_dict}
            cols = get_col_selection()
            combined_meta = {k: v for k, v in combined_meta.items() if k in cols}
            with open(os.path.join(path, 'processed', f'meta.json'), 'w') as f:
                json.dump(combined_meta, f, indent=3)
                
            st.dump_patches(os.path.join(path, 'processed'), 'patches')
            print_resource_usage()
            import psutil
            current_process = psutil.Process()
            child_processes = current_process.children()
            number_of_child_processes = len(child_processes)
            logger.debug(f'{number_of_child_processes} child processes')
            
            del st
            gc.collect()
        except Exception as e:
            traceback.print_exc()
            if not no_except:
                raise e