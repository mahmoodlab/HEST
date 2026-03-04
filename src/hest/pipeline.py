from __future__ import annotations

import gc
import json
import os
import traceback
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import openslide
from hest.trident_compat import WSI
from loguru import logger
from tqdm import tqdm

from hest.HESTData import (VisiumHDHESTData, XeniumHESTData)
from hest.io.seg_readers import GeojsonCellReader, read_gdf, write_geojson
from hest.readers import read_and_save
from hest.registration import preprocess_cells_xenium
from hest.segmentation.cell_segmenters import (bin_per_cell,
                                               cell_segmenter_factory)
from hest.utils import (ALIGNED_HE_FILENAME, deprecated,
                        find_first_file_endswith, get_col_selection,
                        get_path_from_meta_row,
                        print_resource_usage, plot_xenium_align_qc)


@deprecated
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

@deprecated
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
                    write_geojson(warped_cells, os.path.join(path, 'processed', f'he_cell_seg.geojson'))
                    write_geojson(warped_nuclei, os.path.join(path, 'processed', f'he_nucleus_seg.geojson'))
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
                plot_xenium_align_qc(st.wsi, './', st.transcript_df, st.get_shapes('tenx_nucleus', 'he').shapes)
            
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