from __future__ import annotations

import json
import math
import os
import shutil
from typing import Union
import zipfile
from abc import abstractmethod

import numpy as np
import pandas as pd
from hestcore.segmentation import get_path_relative
from loguru import logger

from hest.HESTData import (HESTData, STHESTData, VisiumHDHESTData,
                           VisiumHESTData, XeniumHESTData)
from hest.io.seg_readers import XeniumParquetCellReader, read_gdf
from hest.LazyShapes import LazyShapes
from hest.segmentation.cell_segmenters import segment_cellvit
from hest.utils import (SpotPacking, align_xenium_df, check_arg,
                        find_biggest_img,
                        find_first_file_endswith,
                        find_pixel_size_from_spot_coords,
                        get_path_from_meta_row, helper_mex, load_wsi,
                        metric_file_do_dict, read_xenium_alignment,
                        register_downscale_img, verify_paths)

LOCAL = False
if LOCAL:
    from .custom_readers import (GSE167096_to_adata, GSE180128_to_adata,
                                 GSE203165_to_adata, GSE217828_to_adata,
                                 GSE234047_to_adata, GSE238145_to_adata,
                                 align_dev_human_heart,
                                 align_eval_qual_dataset, align_her2,
                                 align_ST_counts_with_transform,
                                 raw_count_to_adata, raw_counts_to_pixel, ADT_to_adata, 
                                 GSE144239_to_adata, colon_atlas_to_adata, heart_atlas_to_adata)


class Reader:
    """ ST/H&E reader """
    
    def auto_read(self, path: str, **read_kwargs) -> HESTData:
        """
        Automatically detect the file names and determine a reading strategy based on the
        detected files. For more control on the reading process, consider using `read()` instead

        Args:
            path (st): path to the directory containing all the necessary files

        Returns:
            HESTData: STObject that was read
        """
        import scanpy as sc
        
        verify_paths([path])
        
        hest_object = self._auto_read(path, **read_kwargs)
        
        if len(hest_object.adata) > 0:          
            hest_object.adata.var["mito"] = hest_object.adata.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(hest_object.adata, qc_vars=["mito"], inplace=True)
        
        return hest_object
    
    @abstractmethod
    def _auto_read(self, path, **read_kwargs) -> HESTData:
        pass
  
    @abstractmethod
    def read(self, **options) -> HESTData:
        pass


def read_visium_positions_old(tissue_position_list_path):
    tissue_positions = pd.read_csv(tissue_position_list_path, header=None, sep=",", na_filter=False, index_col=0)
    
    tissue_positions = tissue_positions.rename(columns={1: "in_tissue", # in_tissue: 1 if spot is captured in tissue region, 0 otherwise
                                    2: "array_row", # spot row index
                                    3: "array_col", # spot column index
                                    4: "pxl_row_in_fullres", # spot x coordinate in image pixel
                                    5: "pxl_col_in_fullres"}) # spot y coordinate in image pixel
    
    return tissue_positions


class VisiumHDReader(Reader):
    """10x Genomics Visium-HD reader"""
        
    def _auto_read(self, path, **read_kwargs) -> VisiumHDHESTData:
        img_filename = find_biggest_img(path)
        
        square_16um_path = find_first_file_endswith(path, 'square_016um')
        if square_16um_path is None:
            square_16um_path = find_first_file_endswith(os.path.join(path, 'binned_outputs'), 'square_016um')
            
        square_2um_path = find_first_file_endswith(path, 'square_002um', anywhere=True)
        
        metrics_path = find_first_file_endswith(path, 'metrics_summary.csv')
        
        st_object = self.read(
            img_path=os.path.join(path, img_filename),
            square_16um_path=square_16um_path,
            metrics_path=metrics_path,
            square_2um_path=square_2um_path,
            **read_kwargs
        )
        
        return st_object
        
    
    def read(
        self, 
        img_path: str, 
        square_16um_path: str, 
        metrics_path: str = None,
        square_2um_path: str = None,
        use_dask = False
    ) -> VisiumHDHESTData:
        import scanpy as sc
        
        self.square_2um_path = square_2um_path
        
        print("Loading the WSI... (can be slow for large images)")
        img, pixel_size_embedded = load_wsi(img_path)
        
        spatial_path = find_first_file_endswith(square_16um_path, 'spatial')
        tissue_positions_path = find_first_file_endswith(spatial_path, 'tissue_positions.parquet')
        filtered_bc_matrix_path = find_first_file_endswith(square_16um_path, 'filtered_feature_bc_matrix.h5')
        
        tissue_positions = pd.read_parquet(tissue_positions_path)
        tissue_positions.index = tissue_positions['barcode']
        tissue_positions = tissue_positions.drop('barcode', axis=1)

        print("Loading spot counts...") 
        adata = sc.read_10x_h5(filtered_bc_matrix_path)
        
        aligned_spots = pd.merge(adata.obs, tissue_positions, how='inner', left_index=True, right_index=True)
        adata.obs = aligned_spots
        adata.obsm['spatial'] = adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
        
        meta = {}
        if metrics_path is not None:
            meta = metric_file_do_dict(metrics_path) 
            
            
        pixel_size, _ = find_pixel_size_from_spot_coords(adata.obs, inter_spot_dist=16, packing=SpotPacking.GRID_PACKING)


        pooled_adata = pool_bins_visiumhd(adata, pixel_size, dst_bin_size_um=128, src_bin_size_um=16)
            
        meta['pixel_size_um_embedded'] = pixel_size_embedded
        meta['pixel_size_um_estimated'] = pixel_size
        meta['spots_under_tissue'] = len(pooled_adata.obs)
            
        register_downscale_img(pooled_adata, img, pixel_size, spot_size=128)
        
        
        return VisiumHDHESTData(pooled_adata, img, meta['pixel_size_um_estimated'], meta)
        
        
class VisiumReader(Reader):
    """10x Genomics Visium reader"""
        
    
    def _auto_read(self, path, **read_kwargs) -> VisiumHESTData:
        import scanpy as sc
        
        custom_adata = None
        img_filename = find_biggest_img(path)
        
        tissue_positions_path = find_first_file_endswith(path, 'tissue_positions_list.csv')
        if tissue_positions_path is None:
            tissue_positions_path = find_first_file_endswith(path, 'tissue_positions.csv')
        scalefactors_path = find_first_file_endswith(path, 'scalefactors_json.json')
        hires_path = find_first_file_endswith(path, 'tissue_hires_image.png')
        lowres_path = find_first_file_endswith(path, 'tissue_lowres_image.png')
        spatial_coord_path = find_first_file_endswith(path, 'spatial')
        raw_count_path = find_first_file_endswith(path, 'raw_count.txt')
        if spatial_coord_path is None and (tissue_positions_path is not None or \
                scalefactors_path is not None or hires_path is not None or \
                lowres_path is not None or spatial_coord_path is not None):
            os.makedirs(os.path.join(path, 'spatial'), exist_ok=True)
            spatial_coord_path = find_first_file_endswith(path, 'spatial')
        
        if tissue_positions_path is not None:
            shutil.move(tissue_positions_path, spatial_coord_path)
        if scalefactors_path is not None:
            shutil.move(scalefactors_path, spatial_coord_path)
        if hires_path is not None:
            shutil.move(hires_path, spatial_coord_path)
        if lowres_path is not None:
            shutil.move(lowres_path, spatial_coord_path)
        
            
        filtered_feature_path = find_first_file_endswith(path, 'filtered_feature_bc_matrix.h5')
        raw_feature_path = find_first_file_endswith(path, 'raw_feature_bc_matrix.h5')
        alignment_path = find_first_file_endswith(path, 'alignment_file.json')
        if alignment_path is None:
            alignment_path = find_first_file_endswith(path, 'alignment.json')
        if alignment_path is None:
            alignment_path = find_first_file_endswith(path, 'alignment', anywhere=True)
        if alignment_path is None and os.path.exists(os.path.join(path, 'spatial')):
            alignment_path = find_first_file_endswith(os.path.join(path, 'spatial'), 'autoalignment.json')
        if alignment_path is None:
            json_path = find_first_file_endswith(path, '.json')
            if json_path is not None:
                f = open(json_path)
                meta = json.load(f)
                if 'oligo' in meta:
                    alignment_path = json_path
        mex_path = find_first_file_endswith(path, 'mex')
        
        mtx_path = find_first_file_endswith(path, 'matrix.mtx.gz')
        mtx_path = mtx_path if mtx_path is not None else  find_first_file_endswith(path, 'matrix.mtx')
        features_path = find_first_file_endswith(path, 'features.tsv.gz')
        features_path = features_path if features_path is not None else  find_first_file_endswith(path, 'features.tsv')
        barcodes_path = find_first_file_endswith(path, 'barcodes.tsv.gz')
        barcodes_path = barcodes_path if barcodes_path is not None else  find_first_file_endswith(path, 'barcodes.tsv')
        if mex_path is None and (mtx_path is not None or features_path is not None or barcodes_path is not None):
            os.makedirs(os.path.join(path, 'mex'), exist_ok=True)
            mex_path = find_first_file_endswith(path, 'mex')
            shutil.move(mtx_path, mex_path)
            shutil.move(features_path, mex_path)
            shutil.move(barcodes_path, mex_path)
        
        if "Comprehensive Atlas of the Mouse Urinary Bladder" in path:
            custom_adata = GSE180128_to_adata(path)
            
        if "Spatial Transcriptomics of human fetal liver"  in path:
            custom_adata = GSE167096_to_adata(path)
            
        if 'Spatial sequencing of Foreign body granuloma' in path:
            custom_adata = GSE203165_to_adata(path)
            
            
        autoalign = 'auto'
            
        if 'YAP Drives Assembly of a Spatially Colocalized Cellular Triad Required for Heart Renewal' in path:
            custom_adata = GSE217828_to_adata(path)
            autoalign = 'never'
            
        if 'Spatiotemporal mapping of immune and stem cell dysregulation after volumetric muscle loss' in path:
            my_path = find_first_file_endswith(path, '.h5ad')
            custom_adata = sc.read_h5ad(my_path)
            
        if 'The neurons that restore walking after paralysis [spatial transcriptomics]' in path:
            my_path = find_first_file_endswith(path, '.h5ad')
            custom_adata = sc.read_h5ad(my_path)           
            
        if 'GENE EXPRESSION WITHIN A HUMAN CHOROIDAL NEOVASCULAR MEMBRANE USING SPATIAL TRANSCRIPTOMICS' in path:
            custom_adata = GSE234047_to_adata(path)
            
        if 'Batf3-dendritic cells and 4-1BB-4-1BB ligand axis are required at the effector phase within the tumor microenvironment for PD-1-PD-L1 blockade efficacy' in path:
            custom_adata = GSE238145_to_adata(path)
            
        if 'Spatially resolved multiomics of human cardiac niches' in path:
            custom_adata = heart_atlas_to_adata(path)
            autoalign = 'never'
            
        if 'COLON MAP: Colon Molecular Atlas Project' in path:
            custom_adata = colon_atlas_to_adata(path)
            autoalign = 'auto'
            
        if raw_count_path is not None:
            custom_adata = raw_count_to_adata(raw_count_path)
            
        seurat_h5_path = find_first_file_endswith(path, 'seurat.h5ad')
        
        if img_filename is None:
            raise Exception(f"Couldn't detect an image in the directory {path}")
        
        metric_file_path = find_first_file_endswith(path, 'metrics_summary.csv')
        
        st_object = self.read(
            filtered_bc_matrix_path=filtered_feature_path,
            raw_bc_matrix_path=raw_feature_path,
            spatial_coord_path=spatial_coord_path,
            img_path=os.path.join(path, img_filename),
            alignment_file_path=alignment_path,
            mex_path=mex_path,
            scanpy_h5_path=seurat_h5_path,
            metric_file_path=metric_file_path,
            custom_adata=custom_adata,
            autoalign=autoalign,
            save_autoalign=True,
            **read_kwargs
        )
        
        return st_object        
    
    
    def read(self,
        img_path: str,
        filtered_bc_matrix_path: str = None,
        raw_bc_matrix_path: str = None,
        spatial_coord_path: str = None,
        alignment_file_path: str = None, 
        mex_path: str = None,
        scanpy_h5_path: str = None,
        metric_file_path: str = None,
        custom_adata: sc.AnnData = None, # type: ignore
        autoalign: bool = 'auto',
        save_autoalign: bool = False
    ) -> VisiumHESTData:
        """read 10x visium with its associated image
        
        requires a full resolution image and a gene expression file, only one of the gene expression files will be used in this order:
        - filtered_bc_matrix_path
        - raw_bc_matrix_path
        - mex_path
        - custom_adata
        - scanpy_h5_path
        
        
        Args:
            img_path (str): path to the full resolution image
            filtered_bc_matrix_path (str, optional): path to the filtered_feature_bc_matrix.h5. Defaults to None.
            raw_bc_matrix_path (str, optional): path to the raw_feature_bc_matrix.h5. Defaults to None.
            spatial_coord_path (str, optional): path to the spatial/ folder containing either a tissue_positions.csv or a tissue_position_list.csv (is such folder exists). Defaults to None.
            alignment_file_path (str, optional): path to an alignment file (if exists). Defaults to None.
            mex_path (str, optional): path to a folder containing three files ending with barcode.tsv(.gz), features.tsv(.gz), matrix.mtx(.gz). Defaults to None.
            scanpy_h5_path (str, optional): path to a scanpy formated .h5ad. Defaults to None.
            metric_file_path (str, optional): path to a metrics_summary.csv file. Defaults to None.
            custom_adata (sc.AnnData, optional): a scanpy spatial AnnData object. Defaults to None.
            autoalign (str, optional): whenever to use an automated object detector to align the spots based on the fiducials in the full res image. Defaults to 'auto'.
                
                - `auto`: use the autoaligner if no tissue_positions are detected in the spatial folder and if an alignment file is not passed
                
                - `always`: force autoalignment
                
                - `never`: do not use autalignment
            save_autoalign (bool, optional): whenever to save the autoalignment file and visualization plot in a spatial/ folder. Defaults to False.


        Raises:
            ValueError: on invalid arguments

        Returns:
            VisiumHESTData: visium spatial data with spots aligned based on the provided arguments
        """
        import scanpy as sc
        
        print('alignment file is ', alignment_file_path)
        
        check_arg(autoalign, 'autoalign', ['always', 'never', 'auto'])

        if filtered_bc_matrix_path is not None:
            adata = sc.read_10x_h5(filtered_bc_matrix_path)
        elif raw_bc_matrix_path is not None:
            adata = sc.read_10x_h5(raw_bc_matrix_path)
        elif mex_path is not None:
            helper_mex(mex_path, 'barcodes.tsv.gz')
            helper_mex(mex_path, 'features.tsv.gz')
            helper_mex(mex_path, 'matrix.mtx.gz')
                
            adata = sc.read_10x_mtx(mex_path)
        elif custom_adata is not None:
            adata = custom_adata
        elif scanpy_h5_path is not None:
            adata = sc.read_h5ad(scanpy_h5_path)
        else:
            raise ValueError(f"Couldn't find gene expressions, make sure to provide one of the following: filtered_bc_matrix.h5, mex folder, raw_bc_matrix_path, custom_adata, scanpy_h5_path")

        adata.var_names_make_unique()
        print(adata)

        wsi, pixel_size_embedded = load_wsi(img_path)
        
        
        print('trim the barcodes')
        adata.obs.index = [idx[:18] for idx in adata.obs.index]
        if not adata.obs.index[0].endswith('-1'):
            print("barcode don't end with -1 !")
            adata.obs.index = [idx + '-1' for idx in adata.obs.index]

               
        tissue_positions_path = find_first_file_endswith(spatial_coord_path, 'tissue_positions.csv', exclude='aligned_tissue_positions.csv')
        tissue_position_list_path = find_first_file_endswith(spatial_coord_path, 'tissue_positions_list.csv')
        
        tissue_position_exists = tissue_positions_path is not None or tissue_position_list_path is not None
            
        if autoalign == 'always' or (not tissue_position_exists and alignment_file_path is None and autoalign != 'never'):
            if (not tissue_position_exists and alignment_file_path is None):
                print('no tissue_positions_list.csv/tissue_positions.csv or alignment file found')
            print('attempt fiducial auto alignment...')
            from hest.autoalign import autoalign_visium

            os.makedirs(os.path.join(os.path.dirname(img_path), 'spatial'), exist_ok=True)
            autoalign_save_dir = None
            if save_autoalign:
                autoalign_save_dir = os.path.join(os.path.dirname(img_path), 'spatial')
            align_json = autoalign_visium(wsi.numpy(), autoalign_save_dir)
            spatial_aligned = self._alignment_file_to_tissue_positions('', adata, align_json)
        
        elif tissue_position_exists:
            # SpaceRanger >= 2.0
            if tissue_positions_path is not None:
                tissue_positions = pd.read_csv(tissue_positions_path, sep=",", na_filter=False, index_col=0)
            # SpaceRanger < 2.0
            elif tissue_position_list_path is not None:
                tissue_positions = read_visium_positions_old(tissue_position_list_path)

            tissue_positions.index = [idx[:18] for idx in tissue_positions.index]
            spatial_aligned = self._align_tissue_positions(
                alignment_file_path, 
                tissue_positions, 
                adata
            )

            assert np.array_equal(spatial_aligned.index, adata.obs.index)

        elif alignment_file_path is not None:
            spatial_aligned = self._alignment_file_to_tissue_positions(alignment_file_path, adata)

        else:
            spatial_aligned = adata.obs[['in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]

        col1 = spatial_aligned['pxl_col_in_fullres'].values
        col2 = spatial_aligned['pxl_row_in_fullres'].values
        
        matrix = np.vstack((col1, col2)).T
        
        adata.obsm['spatial'] = matrix


        pixel_size, spot_estimate_dist = find_pixel_size_from_spot_coords(spatial_aligned)

        adata.obs = spatial_aligned
            
        register_downscale_img(adata, wsi, pixel_size)
        
        dict = {}
        if metric_file_path is not None:
            dict = metric_file_do_dict(metric_file_path)
        
        width, height = wsi.get_dimensions()
        dict['pixel_size_um_embedded'] = pixel_size_embedded
        dict['pixel_size_um_estimated'] = pixel_size
        dict['fullres_height'] = height
        dict['fullres_width'] = width
        dict['spots_under_tissue'] = len(adata.obs)
        dict['spot_estimate_dist'] = int(spot_estimate_dist)
        dict['spot_diameter'] = 55.
        dict['inter_spot_dist'] = 100.
        

        return VisiumHESTData(adata, wsi, dict['pixel_size_um_estimated'], dict)
    

    def _alignment_file_to_df(self, path, alignment_json=None):
        if alignment_json is not None:
            data = alignment_json
        else:
            f = open(path)
            data = json.load(f)
        
        df = pd.DataFrame(data['oligo'])
        
        if 'cytAssistInfo' in data:
            transform = np.array(data['cytAssistInfo']['transformImages'])
            transform = np.linalg.inv(transform)
            matrix = np.column_stack((df['imageX'], df['imageY'], np.ones((len(df['imageX']),))))
            matrix = (transform @ matrix.T).T
            df['imageX'] = matrix[:, 0]
            df['imageY'] = matrix[:, 1]


        return df
    

    def _alignment_file_to_tissue_positions(self, alignment_file_path, adata, alignment_json=None):
        alignment_df = self._alignment_file_to_df(alignment_file_path, alignment_json)
        alignment_df = alignment_df.rename(columns={
            'tissue': 'in_tissue',
            'row': 'array_row',
            'col': 'array_col',
            'imageX': 'pxl_col_in_fullres',
            'imageY': 'pxl_row_in_fullres'
        })
        alignment_df['in_tissue'] = [True for _ in range(len(alignment_df))]

        spatial_aligned = self._find_visium_slide_version(alignment_df, adata)
        return spatial_aligned
    

    def _find_alignment_barcodes(self, alignment_df: str, barcode_path: str) -> pd.DataFrame:
        barcode_coords = pd.read_csv(barcode_path, sep='\t', header=None)
        barcode_coords = barcode_coords.rename(columns={
            0: 'barcode',
            1: 'array_col',
            2: 'array_row'
        })
        barcode_coords['barcode'] += '-1'
        
        # space rangers provided barcode coords are 1 indexed whereas alignment file are 0 indexed
        barcode_coords['array_col'] -= 1
        barcode_coords['array_row'] -= 1

        spatial_aligned = pd.merge(alignment_df, barcode_coords, on=['array_row', 'array_col'], how='inner')

        spatial_aligned.index = spatial_aligned['barcode']

        spatial_aligned = spatial_aligned[['in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]

        return spatial_aligned    
    

    def _find_visium_slide_version(self, alignment_df: str, adata: sc.AnnData) -> str: # type: ignore
        highest_nb_match = -1
        barcode_dir = get_path_relative(__file__, '../../assets/barcode_coords/')
        for barcode_path in os.listdir(barcode_dir):
            spatial_aligned = self._find_alignment_barcodes(alignment_df, os.path.join(barcode_dir, barcode_path))
            nb_match = len(pd.merge(spatial_aligned, adata.obs, left_index=True, right_index=True))
            if nb_match > highest_nb_match:
                highest_nb_match = nb_match
                match_spatial_aligned = spatial_aligned
        
        if highest_nb_match == 0:
            raise Exception(f"Couldn't find a visium slide having the following spot barcodes: {adata.obs.index}")
            
        spatial_aligned = match_spatial_aligned.reindex(adata.obs.index)
        return spatial_aligned
    

    def _align_tissue_positions(
        self,
        alignment_file_path, 
        tissue_positions, 
        adata
    ):
        if alignment_file_path is not None:

            alignment_df = self._alignment_file_to_df(alignment_file_path)
            
            if len(alignment_df) > 0:
                alignment_df = alignment_df.rename(columns={
                    'row': 'array_row',
                    'col': 'array_col',
                    'imageX': 'pxl_col_in_fullres', # TODO had a problem here for the prostate dataset
                    'imageY': 'pxl_row_in_fullres'
                })
                tissue_positions = tissue_positions.rename(columns={
                    'pxl_col_in_fullres': 'pxl_col_in_fullres_old',
                    'pxl_row_in_fullres': 'pxl_row_in_fullres_old'
                })
            
                alignment_df = alignment_df[['array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]
                tissue_positions['barcode'] = tissue_positions.index
                df_merged = alignment_df.merge(tissue_positions, on=['array_row', 'array_col'], how='inner')
                df_merged.index = df_merged['barcode']
                df_merged = df_merged.drop('barcode', axis=1)
                df_merged = df_merged.loc[adata.obs.index]
                df_merged = df_merged.reindex(adata.obs.index)
                
                col1 = df_merged['pxl_col_in_fullres'].values
                col2 = df_merged['pxl_row_in_fullres'].values
                matrix = (np.vstack((col1, col2))).T
                
                adata.obsm['spatial'] = matrix 
                
                spatial_aligned = df_merged
                
            else:
                col1 = tissue_positions['pxl_col_in_fullres'].values
                col2 = tissue_positions['pxl_row_in_fullres'].values        
                    
                spatial_aligned = tissue_positions.reindex(adata.obs.index)
        else:
            spatial_aligned = tissue_positions
            spatial_aligned = spatial_aligned.loc[adata.obs.index]
        return spatial_aligned
    
    
class STReader(Reader):
    """ Legacy Spatial Transcriptomics reader """
    
    def _auto_read(self, path, **read_kwargs) -> STHESTData:
        packing = SpotPacking.GRID_PACKING
        meta_table_path = None
        custom_adata = None
        inter_spot_dist = 200
        spot_diameter = 100

        img_path = find_biggest_img(path)
        
        for file in os.listdir(path):
            if 'meta' in file:
                meta_table_path = os.path.join(path, file)
                break
        raw_counts_path = None
        for file in os.listdir(path):
            if 'count' in file or 'stdata' in file:
                raw_counts_path = os.path.join(path, file)
                break
            
        spot_coord_path = None
        for file in os.listdir(path):
            if 'spot' in file:
                spot_coord_path = os.path.join(path, file)
                break

        transform_path = None
        for file in os.listdir(path):
            if 'transform' in file:
                transform_path = os.path.join(path, file)
                break
            
        if "Prostate needle biopsies pre- and post-ADT: Count matrices, histological-, and Androgen receptor immunohistochemistry images" in path:
            custom_adata = ADT_to_adata(os.path.join(path, img_path), raw_counts_path)
            
        if "Single Cell and Spatial Analysis of Human Squamous Cell Carcinoma [ST]" in path:
            custom_adata = GSE144239_to_adata(raw_counts_path, spot_coord_path)
            packing = SpotPacking.GRID_PACKING
            inter_spot_dist = 110.
            spot_diameter = 150.
            
            
        if 'A spatiotemporal organ-wide gene expression and cell atlas of the developing human heart' in path:
            exp_name = path.split('/')[-1]
            custom_adata = align_dev_human_heart(raw_counts_path, spot_coord_path, exp_name)
            
        
        if 'The spatial RNA integrity number assay for in situ evaluation of transcriptome quality' in path:
            custom_adata = align_eval_qual_dataset(raw_counts_path, spot_coord_path)    
            
        if 'Spatial deconvolution of HER2-positive breast cancer delineates tumor-associated cell type interactions' in path:
            custom_adata = align_her2(path, raw_counts_path)  
            
        if 'Molecular Atlas of the Adult Mouse Brain' in path:
            spot_diameter = 50.      
            
       
        st_object = self.read(
            meta_table_path, 
            raw_counts_path, 
            os.path.join(path, img_path), 
            spot_coord_path, 
            transform_path,
            packing=packing,
            inter_spot_dist=inter_spot_dist,
            spot_diameter=spot_diameter,
            custom_adata=custom_adata,
            **read_kwargs)
        
        return st_object
    
    
    def read(
        self,
        meta_table_path=None, 
        raw_counts_path=None, 
        img_path=None, 
        spot_coord_path=None,
        transform_path=None, 
        packing: SpotPacking = SpotPacking.GRID_PACKING, 
        spot_diameter=100.,
        inter_spot_dist=200.,
        custom_adata=None
    ) -> STHESTData:
        #raw_counts = pd.read_csv(raw_counts_path, sep='\t')
        img, pixel_size_embedded = load_wsi(img_path)
        
        if custom_adata is not None:
            adata = custom_adata
            matrix = adata.obsm['spatial']
           
        elif transform_path is not None:
            # for "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics"
            adata = align_ST_counts_with_transform(raw_counts_path, transform_path)
            matrix = adata.obsm['spatial']
        else:
            raw_counts = pd.read_csv(raw_counts_path, sep='\t')
            if 'Unnamed: 0' in raw_counts.columns:
                raw_counts.index = raw_counts['Unnamed: 0']
                raw_counts = raw_counts.drop(['Unnamed: 0'], axis=1)
            if meta_table_path is not None:
                import scanpy as sc
                meta = pd.read_csv(meta_table_path, sep='\t', index_col=0)
                merged = pd.merge(meta, raw_counts, left_index=True, right_index=True, how='inner')
                raw_counts = raw_counts.reindex(merged.index)
                raw_counts.index = [idx.split('_')[1] for idx in raw_counts.index]
                adata = sc.AnnData(raw_counts)
                col1 = merged['HE_X'].values
                col2 = merged['HE_Y'].values
                matrix = (np.vstack((col1, col2))).T
            elif spot_coord_path is not None:
                #spot_coord = pd.read_csv(spot_coord_path, sep='\t', index_col=0)
                spot_coord = pd.read_csv(spot_coord_path, sep=',', index_col=0)
                merged = pd.merge(spot_coord, raw_counts, left_index=True, right_index=True, how='inner')
                raw_counts = raw_counts.reindex(merged.index)
                adata = sc.AnnData(raw_counts) 

                col1 = merged['X'].values
                col2 = merged['Y'].values
                    
                matrix = (np.vstack((col1, col2))).T
                
            else:
                matrix = raw_counts_to_pixel(raw_counts, img)
                raw_counts = raw_counts.transpose()
                raw_counts.index = [idx.replace('_', 'x') for idx in raw_counts.index]
                adata = sc.AnnData(raw_counts)
        
        adata.obsm['spatial'] = matrix
        
        # TODO get real pixel size
        my_df = pd.DataFrame(adata.obsm['spatial'], adata.to_df().index, columns=['pxl_col_in_fullres', 'pxl_row_in_fullres'])
        my_df['array_row'] = [round(float(idx.split('x')[0])) for idx in my_df.index]
        my_df['array_col'] = [round(float(idx.split('x')[1])) for idx in my_df.index]
        
        adata.obs['array_row'] = my_df['array_row']
        adata.obs['array_col'] = my_df['array_col']
        adata.obs['pxl_col_in_fullres'] = my_df['pxl_col_in_fullres']
        adata.obs['pxl_row_in_fullres'] = my_df['pxl_row_in_fullres']
        adata.obs['in_tissue'] = [True for _ in range(len(adata.obs))]
        
        # TODO might not be precise if we use round on the column and row
        pixel_size, spot_estimate_dist = find_pixel_size_from_spot_coords(my_df, inter_spot_dist=inter_spot_dist, packing=packing)
        register_downscale_img(adata, img, pixel_size, spot_size=spot_diameter)
        
        dict = {}
        dict['pixel_size_um_embedded'] = pixel_size_embedded
        dict['pixel_size_um_estimated'] = pixel_size
        dict['fullres_height'] = img.shape[0]
        dict['fullres_width'] = img.shape[1]
        dict['spots_under_tissue'] = len(adata.obs)
        dict['spot_estimate_dist'] = int(spot_estimate_dist)
        dict['spot_diameter'] = spot_diameter
        dict['inter_spot_dist'] = inter_spot_dist
        
        print(f"'pixel_size_um_embedded' is {pixel_size_embedded}")
        print(f"'pixel_size_um_estimated' is {pixel_size} estimated by averaging over {spot_estimate_dist} spots")
        print(f"'spots_under_tissue' is {len(adata.obs)}")
        
        # make all the indices the same length (important when saving to h5)
        assert 'x' in adata.obs.index[0]
        adata.obs.index = [idx.split('x')[0].zfill(3) + 'x' + idx.split('x')[1].zfill(3) for idx in adata.obs.index]
        
        return STHESTData(adata, img, dict['pixel_size_um_estimated'], dict)
    
    
class XeniumReader(Reader):
    """ 10x Xenium reader """

    def auto_read(self, path: str, **read_kwargs) -> XeniumHESTData:
        """
        Automatically detect the file names and determine a reading strategy based on the
        detected files. For more control on the reading process, consider using `read()` instead

        auto_read will automatically identify the following files based on their name:
        - a WSI: largest image in `path` (see `find_biggest_img`)
        - an image alignment file: any file ending with 'imagealignment.csv'
            this file contains an affine transformation used to align transcripts, 
            and cells from DAPI to H&E. The transformation must be in the standard Xenium format
        - a transcript file: file named `transcripts.parquet`
        - an experiment file: experiment.xenium
        - a `cell_feature_matrix.h5` file
        - a `cells.parquet` file
        - a `nucleus_boundaries.parquet` file
        - a `cell_boundaries.parquet` file

        Args:
            path (st): path to the directory containing all the necessary files

        Returns:
            XeniumHESTData: STObject that was read
        """
        super().auto_read(path, **read_kwargs)
        
    
    def _auto_read(self, path, **read_kwargs) -> XeniumHESTData:
        img_filename = find_biggest_img(path)
                
        alignment_path = None
        for file in os.listdir(path):
            if file.endswith('imagealignment.csv'):
                alignment_path = os.path.join(path, file)
        alignment_path = read_kwargs.pop('alignment_path', alignment_path)
                
        dapi_path = None
        dapi_path = find_first_file_endswith(
            path,
            'morphology_focus.ome.tif'
        )
        if dapi_path is None:
            dapi_path = find_first_file_endswith(
                os.path.join(path, 'morphology_focus'),
                'morphology_focus_0000.ome.tif'
            )
            
        transcripts_path = os.path.join(path, 'transcripts.parquet')
        if 'transcripts_filename' in read_kwargs:
            transcripts_path = os.path.join(path, read_kwargs.pop('transcripts_filename'))
        
        st_object = self.read(
            img_path=os.path.join(path, img_filename),
            experiment_path=os.path.join(path, 'experiment.xenium'),
            alignment_file_path=alignment_path,
            feature_matrix_path=os.path.join(path, 'cell_feature_matrix.h5'), 
            transcripts_path=transcripts_path,
            cells_path=os.path.join(path, 'cells.parquet'),
            nucleus_bound_path=os.path.join(path, 'nucleus_boundaries.parquet'),
            cell_bound_path=os.path.join(path, 'cell_boundaries.parquet'),
            dapi_path=dapi_path,
            **read_kwargs
        )
        
        return st_object
    
    
    def __xenium_estimate_pixel_size(self, pixel_size_morph, he_to_morph_matrix) -> float:
        # define a line by two points in space and get the scale of the line after transformation
        # to determine the pixel size in the H&E image
        two_points = np.array([[0., 0., 1.], [0., 1., 1.]])
        scaled_points = (he_to_morph_matrix @ two_points.T).T
        
        cart_dist = np.sqrt(
            (scaled_points[0][0] - scaled_points[1][0]) ** 2 
            + 
            (scaled_points[0][1] - scaled_points[1][1]) ** 2
        )
        
        pixel_size_estimated = pixel_size_morph * cart_dist
        return pixel_size_estimated
        
        
    def __load_transcripts(self, transcripts_path, alignment_matrix, pixel_size_morph, use_dask):

        if use_dask:
            import dask.dataframe as dd
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(transcripts_path)
            total_row_groups = parquet_file.num_row_groups
            row_groups_per_partition = max(1, total_row_groups // 30)
            df_transcripts = dd.read_parquet(transcripts_path, split_row_groups=row_groups_per_partition)
        else:
            df_transcripts = pd.read_parquet(transcripts_path)
        
        if alignment_matrix is not None:
            df_transcripts['he_x'] = 0.
            df_transcripts['he_y'] = 0.
            if use_dask:
                df_transcripts = df_transcripts.map_partitions(align_xenium_df,
                    alignment_matrix, 
                    pixel_size_morph, 
                    'x_location',
                    'y_location',
                    meta=df_transcripts._meta,
                )
            else:
                df_transcripts = align_xenium_df(
                    df_transcripts,
                    alignment_matrix, 
                    pixel_size_morph, 
                    'x_location',
                    'y_location',
                )
            df_transcripts['dapi_x'] = df_transcripts['x_location'] / pixel_size_morph
            df_transcripts['dapi_y'] = df_transcripts['y_location'] / pixel_size_morph
        else:
            df_transcripts['he_x'] = df_transcripts['x_location'] / pixel_size_morph
            df_transcripts['he_y'] = df_transcripts['y_location'] / pixel_size_morph
            df_transcripts['dapi_x'] = df_transcripts['x_location'] / pixel_size_morph
            df_transcripts['dapi_y'] = df_transcripts['y_location'] / pixel_size_morph
            
        return df_transcripts
    
    
    def __load_cells(self, feature_matrix_path, cells_path, alignment_matrix, pixel_size_morph, dict):
        import scanpy as sc
        
        cell_adata = sc.read_10x_h5(feature_matrix_path)
        df = pd.read_parquet(cells_path)
        df.set_index(cell_adata.obs_names, inplace=True)
        cell_adata.obs = df.copy()
        
        df_cells = cell_adata.obs[["x_centroid", "y_centroid"]].copy()
        
        if alignment_matrix is not None:
            # convert cell coordinates from um in the morphology image to um in the H&E image
            df_cells = align_xenium_df(df_cells, alignment_matrix, pixel_size_morph, x_key='x_centroid', y_key='y_centroid')
        df_cells['he_x'] = df_cells['x_centroid'] / pixel_size_morph
        df_cells['he_y'] = df_cells['y_centroid'] / pixel_size_morph
        
        cell_adata.obsm["spatial"] = df_cells[['he_x', 'he_y']].to_numpy()
        cell_adata.obs[['he_x', 'he_y']] = df_cells[['he_x', 'he_y']]
        dict['cells_under_tissue'] = len(cell_adata.obs)
        return cell_adata, dict
    
    
    def read(
        self,
        img_path: str,
        experiment_path: str,
        alignment_file_path: str = None,
        feature_matrix_path: str = None, 
        transcripts_path: str = None,
        cells_path: str = None,
        nucleus_bound_path: str = None,
        cell_bound_path: str = None,
        dapi_path = None,
        load_img=True,
        use_dask=False,
        spot_size_um=100.
    ) -> XeniumHESTData:
        """ Read a Xenium sample

        Args:
            img_path (str): path to the WSI
            experiment_path (str): path to a `experiment.xenium` file
            alignment_file_path (str, optional): path to a DAPI->H&E alignment file, None if the H&E is already aligned with the DAPI. Defaults to None.
            feature_matrix_path (str, optional): path to a `cell_feature_matrix.h5`. Defaults to None.
            transcripts_path (str, optional): path to a transcripts.parquet, None to not load the transcripts. Defaults to None.
            cells_path (str, optional): path to a `cells.parquet` file, None to not load the cells. Defaults to None.
            nucleus_bound_path (str, optional): path to a `nucleus_boundaries.parquet` file. Defaults to None.
            cell_bound_path (str, optional): path to a `cell_boundaries` file. Defaults to None.
            dapi_path (_type_, optional): path to a `morphology_focus_0000.ome.tif`/`morphology_focus.ome.tif` file. Defaults to None.
            load_img (bool, optional): whenever to load the WSI. Defaults to True.
            use_dask (bool, optional): whenever to load the transcript dataframe with DASK (recommended if the transcript dataframe does not fit into the RAM). Defaults to False.
            spot_size_um (float, optional): transcripts are pooled into squares of spot_size_um x spot_size_um mirometers and then stored in `HESTData.adata`

        Returns:
            XeniumHESTData: Xenium sample
        """
            
        if load_img:
            print("Loading the WSI... (can be slow for large images)")
            img, pixel_size_embedded = load_wsi(img_path)
        else:
            img, pixel_size_embedded = np.zeros((1, 1, 3)), None
        
        dict = {}
        dict['pixel_size_um_embedded'] = pixel_size_embedded
        
        with open(experiment_path) as f:
            dict_exp = json.load(f)
            pixel_size_morph = dict_exp['pixel_size']
        dict = {**dict, **dict_exp}
        
        shapes = []


        alignment_matrix = read_xenium_alignment(alignment_file_path) if alignment_file_path else None
        dict['pixel_size_um_estimated'] = self.__xenium_estimate_pixel_size(pixel_size_morph, alignment_matrix)
        if cell_bound_path is not None:
            shapes.append(LazyShapes(cell_bound_path, 'tenx_cell', 'dapi', reader=XeniumParquetCellReader, reader_kwargs={'pixel_size_morph': pixel_size_morph}))
            if alignment_matrix is not None:
                shapes.append(LazyShapes(cell_bound_path, 'tenx_cell', 'he', reader=XeniumParquetCellReader, reader_kwargs={'pixel_size_morph': pixel_size_morph, 'alignment_matrix': alignment_matrix}))

        if nucleus_bound_path is not None:
            shapes.append(LazyShapes(nucleus_bound_path, 'tenx_nucleus', 'dapi', reader=XeniumParquetCellReader, reader_kwargs={'pixel_size_morph': pixel_size_morph}))
            if alignment_matrix is not None:
                shapes.append(LazyShapes(nucleus_bound_path, 'tenx_nucleus', 'he', reader=XeniumParquetCellReader, reader_kwargs={'pixel_size_morph': pixel_size_morph, 'alignment_matrix': alignment_matrix}))

        
        if transcripts_path is not None:
            print('Loading transcripts...')
            transcript_df = self.__load_transcripts(transcripts_path, alignment_matrix, pixel_size_morph, use_dask)
                    
            print("Pooling xenium transcripts in pseudo-visium spots...")
            adata = pool_transcripts_xenium(
                transcript_df, 
                dict['pixel_size_um_estimated'], 
                key_x='he_x',
                key_y='he_y',
                spot_size_um=spot_size_um
            )
            
            dict['spot_diameter'] = spot_size_um
            dict['inter_spot_dist'] = spot_size_um
            dict['spots_under_tissue'] = len(adata.obs)
            
        else:
            import scanpy as sc
            adata = sc.AnnData()
            adata.obsm['spatial'] = np.array([])
            transcript_df = None

        register_downscale_img(adata, img, dict['pixel_size_um_estimated'])
            
        cell_adata = None
        if feature_matrix_path is not None and cells_path is not None:
            print('Reading cells...')
            cell_adata, dict = self.__load_cells(feature_matrix_path, cells_path, alignment_matrix, pixel_size_morph, dict)

            
        st_object = XeniumHESTData(
            adata, 
            img, 
            dict['pixel_size_um_estimated'], 
            dict, 
            transcript_df=transcript_df,
            cell_adata=cell_adata,
            shapes=shapes,
            dapi_path=dapi_path,
            alignment_file_path=alignment_file_path
        )
        return st_object
    

def reader_factory(path: str) -> Reader:
    """For internal use, determine the reader based on the path"""
    path = path.lower()
    if 'visium-hd' in path:
        return VisiumHDReader()
    elif 'visium' in path:
        return VisiumReader()
    elif 'xenium' in path:
        return XeniumReader()
    elif 'st' in path:
        return STReader()
    else:
        raise NotImplementedError('')
        
def read_and_save(path: str, save_plots=True, pyramidal=True, bigtiff=False, plot_pxl_size=False, save_img=True, segment_tissue=False, read_kwargs={}, save_kwargs={}, segment_kwargs={}):
    """For internal use, determine the appropriate reader based on the raw data path, and
    automatically process the data at that location, then the processed files are dumped
    to processed/

    Args:
        path (str): path of the raw data
        save_plots (bool, optional): whenever to save the spatial plots. Defaults to True.
        pyramidal (bool, optional): whenever to save as pyramidal. Defaults to True.
    """
    print(f'Reading from {path}...')
    reader = reader_factory(path)
    st_object = reader.auto_read(path, **read_kwargs)
    logger.info('Loaded object:')
    print(st_object)
    if segment_tissue:
        logger.info('Segment tissue')
        st_object.segment_tissue(**segment_kwargs)
    save_path = os.path.join(path, 'processed')
    os.makedirs(save_path, exist_ok=True)
    st_object.save(save_path, pyramidal=pyramidal, bigtiff=bigtiff, plot_pxl_size=plot_pxl_size, save_img=save_img, **save_kwargs)
    if save_plots:
        st_object.save_spatial_plot(save_path)
    return st_object
        
def pool_transcripts_xenium(
    df: Union[pd.DataFrame, dd.DataFrame], 
    pixel_size_he: float,
    spot_size_um=100.,
    key_x='he_x',
    key_y='he_y',
) -> sc.AnnData: # type: ignore
    """ Pool a xenium transcript dataframe by square spots of `spot_size_um` micrometers.

    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): xenium transcipts dataframe containing columns:
            - 'he_x' and 'he_y' indicating the pixel coordinates of each transcripts in the morphology image
            - 'feature_name' indicating the transcript name
        pixel_size_he (float): pixel_size in um on the he image
        spot_size_um: pooling rectangle width in um
        key_x: column name of pixel x coordinate of each transcript in `df`
        key_y: column name of pixel y coordinate of each transcript in `df`
        

    Returns:
        sc.AnnData: AnnData object, each row in .obs represents a bin, each row in `.X` represents the sum of transcripts within that bin. Center coordinates of each bin (in pixel on WSI) are in adata.obsm['spatial']
    """
    import scanpy as sc
    import dask.dataframe as dd

    y_max = df[key_y].max()
    y_min = df[key_y].min()
    x_max = df[key_x].max()
    x_min = df[key_x].min()
    
    m = ((y_max - y_min) / (spot_size_um / pixel_size_he))
    n = ((x_max - x_min) / (spot_size_um / pixel_size_he))

    if isinstance(df, dd.DataFrame):
        m = m.compute()
        n = n.compute()

    m = math.ceil(m)
    n = math.ceil(n)
    
    features = df['feature_name'].unique()
    if isinstance(df, dd.DataFrame):
        features = features.compute()
    
    spot_grid = pd.DataFrame(0, index=range(m * n), columns=features)
    

    # a is the row and b is the column in the pseudo visium grid
    a = np.floor((df[key_x] - x_min) / (spot_size_um / pixel_size_he)).astype(int)
    b = np.floor((df[key_y] - y_min) / (spot_size_um / pixel_size_he)).astype(int)
    
    c = b * n + a
    features = df['feature_name']
    
    cols = spot_grid.columns.get_indexer(features)
    
    ## use dask for this parts
    spot_grid_np = spot_grid.values.astype(np.uint16)
    np.add.at(spot_grid_np, (c, cols), 1)
    
    
    if isinstance(spot_grid.columns.values[0], bytes):
        spot_grid.columns = [i.decode('utf-8') for i in spot_grid.columns]
    

    expression_df = pd.DataFrame(spot_grid_np, columns=spot_grid.columns)
    
    coord_df = expression_df.copy()
    if isinstance(df, dd.DataFrame):
        x_min = x_min.compute()
        y_min = y_min.compute()

    coord_df['x'] = x_min + (coord_df.index % n) * (spot_size_um / pixel_size_he) + ((spot_size_um / 2) / pixel_size_he)
    coord_df['y'] = y_min + np.floor(coord_df.index / n) * (spot_size_um / pixel_size_he) + ((spot_size_um / 2) / pixel_size_he)
    coord_df = coord_df[['x', 'y']]
    
    expression_df.index = [str(i) for i in expression_df.index]
    
    adata = sc.AnnData(expression_df)
    adata.var_names = adata.var_names.astype(str)
    adata.obsm['spatial'] = coord_df[['x', 'y']].values
    adata.obs['in_tissue'] = True
    adata.obs['pxl_col_in_fullres'] = coord_df['x'].values
    adata.obs['pxl_row_in_fullres'] = coord_df['y'].values
    adata.obs['array_col'] = np.arange(len(adata.obs)) % n
    adata.obs['array_row'] = np.arange(len(adata.obs)) // n
    adata.obs.index = [str(row).zfill(3) + 'x' + str(col).zfill(3) for row, col in  zip(adata.obs['array_row'], adata.obs['array_col'])]
    sc.pp.filter_cells(adata, min_counts=1)
    
    return adata


def pool_bins_visiumhd(adata: sc.AnnData, pixel_size: float, dst_bin_size_um=128, src_bin_size_um: Literal[2, 8, 16]=16, chunk_len=50000) -> sc.AnnData: # type: ignore
    """ Pool a Visium HD (with a source resolution of `src_bin_size_um`) by square spots of `spot_size_um` micrometers.

    Args:
        adata (sc.AnnData): adata containing spot center coordiniates in `pxl_row_in_fullres` and `pxl_col_in_fullres`
        pixel_size (float): pixel size of the WSI in um/px
        dst_bin_size_um (int, optional): target bin size in um. Defaults to 128.
        src_bin_size_um (Literal[2, 8, 16], optional): bin size of `adata` in um. Defaults to 16.
        chunk_len (int, optional): chunk size when binning a larger than RAM `adata` (this is for RAM optimization only). Defaults to 50000.

    Returns:
        sc.AnnData: AnnData object, each row in .obs represents a bin, each row in `.X` represents the sum of visium-hd sub-bins (of size `src_bin_size_um`) within that larger bin (of size `dst_bin_size_um`). Center coordinates of each bin (in pixel on WSI) are in adata.obsm['spatial']
    """
    import scanpy as sc

    if src_bin_size_um >= dst_bin_size_um:
        raise ValueError("dst_bin_size_um needs to be larger than src_bin_size_um")
    
    y_max = adata.obs['pxl_row_in_fullres'].max()
    y_min = adata.obs['pxl_row_in_fullres'].min()
    x_max = adata.obs['pxl_col_in_fullres'].max()
    x_min = adata.obs['pxl_col_in_fullres'].min()
    
    # pxl_col_in_fullres and pxl_row_in_fullres refer to the center of each spot so we add + src_bin_size_um / pixel_size
    grid_height_pxl = y_max - y_min + src_bin_size_um / pixel_size
    grid_width_pxl = x_max - x_min + src_bin_size_um / pixel_size
    dst_bin_pxl_size = dst_bin_size_um / pixel_size
    src_bin_pxl_size = src_bin_size_um / pixel_size

    m = math.ceil(grid_height_pxl / dst_bin_pxl_size)
    n = math.ceil(grid_width_pxl / dst_bin_pxl_size)

    features = adata.var_names
    
    spot_grid = pd.DataFrame(0, index=range(m * n), columns=features)
    
    # a is the row and b is the column in the binned grid
    a = np.floor((adata.obs['pxl_col_in_fullres'] - x_min + src_bin_pxl_size // 2) / dst_bin_pxl_size).astype(int)
    b = np.floor((adata.obs['pxl_row_in_fullres'] - y_min + src_bin_pxl_size // 2) / dst_bin_pxl_size).astype(int)
    
    c = b * n + a
    c = np.array(c)
    spot_grid_np = spot_grid.values.astype(np.float32)

    nb_chunks = int(np.ceil(len(c) / chunk_len))

    for i in range(nb_chunks):
        start, end = i * chunk_len, min((i + 1) * chunk_len, len(c))
        spot_grid_np[c[start:end]] += adata.X[start:end]

    
    expression_df = pd.DataFrame(spot_grid_np, columns=spot_grid.columns)
    
    row_sums = expression_df.sum(axis=1)

    # Filter rows where the sum is not equal to zero
    expression_df = expression_df[row_sums > 1e-8]
    
    
    #coord_df = expression_df.copy()
    pos_x = x_min + (expression_df.index % n) * dst_bin_pxl_size + dst_bin_pxl_size // 2
    pos_y = y_min + np.floor(expression_df.index / n) * dst_bin_pxl_size + dst_bin_pxl_size // 2
    
    
    adata = sc.AnnData(expression_df) # type: ignore
    adata.obsm['spatial'] = np.column_stack((pos_x, pos_y))
    adata.obs['in_tissue'] = [True for _ in range(len(adata.obs))]
    adata.obs['pxl_col_in_fullres'] = pos_x
    adata.obs['pxl_row_in_fullres'] = pos_y
    adata.obs['array_col'] = np.arange(len(adata.obs)) % n
    adata.obs['array_row'] = np.arange(len(adata.obs)) // n
    adata.obs.index = [str(row).zfill(4) + 'x' + str(col).zfill(4) for row, col in  zip(adata.obs['array_row'], adata.obs['array_col'])]
    
    return adata

    

def _process_cellvit(row, **cellvit_kwargs):
    path = get_path_from_meta_row(row)
    wsi_path = os.path.join(path, 'processed', 'aligned_fullres_HE.tif')
    with open(os.path.join(path, 'processed', 'metrics.json')) as f:
        meta = json.load(f)
    src_cell_path = segment_cellvit(wsi_path, row['id'], meta['pixel_size_um_estimated'], **cellvit_kwargs)
    dst_cell_path = os.path.join(path, 'processed', 'cellvit_seg.geojson')
    shutil.copy(src_cell_path, dst_cell_path)
    
    gdf = read_gdf(dst_cell_path)
    dst_parquet_path = os.path.join(path, 'processed', row['id'] + '_cellvit_seg.parquet')
    gdf.to_parquet(dst_parquet_path)
    
    archive_path = os.path.join(path, 'processed', f'cellvit_seg.zip')
    
    id = row['id']
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(os.path.join(path, 'processed', f'cellvit_seg.geojson'), f'{id}_cellvit_seg.geojson')
    
        
def process_meta_df_cellvit(meta_df, cellvit_kwargs={'gpu_ids': [0, 1], 'batch_size': 4}):
    for _, row in meta_df.iterrows():
        _process_cellvit(row, **cellvit_kwargs)
    
