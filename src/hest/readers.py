from abc import abstractmethod
from dataclasses import dataclass
import os
import json
from typing import Tuple, Dict
import numpy as np
import scanpy as sc
import pandas as pd
import shutil
import traceback
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from src.hest.helpers import _find_first_file_endswith, _helper_mex, _txt_matrix_to_adata, \
    _raw_count_to_adata, GSE234047_to_h5, GSE180128_to_h5, GSE203165_to_adata, _load_image, \
    _align_tissue_positions, _alignment_file_to_tissue_positions, autoalign_with_fiducials, \
    _find_pixel_size_from_spot_coords, _register_downscale_img, _metric_file_do_dict, \
    align_ST_counts_with_transform, raw_counts_to_pixel, _find_biggest_img, write_10X_h5, _save_scalefactors, \
    _plot_verify_pixel_size, xenium_to_pseudo_visium, write_wsi, GSE167096_to_adata, GSE217828_to_custom, \
    align_dev_human_heart, align_her2, align_eval_qual_dataset, _get_path_from_meta_row, SpotPacking


ALIGNED_HE_FILENAME = 'aligned_fullres_HE.tif'
    
    
"""class HESTMeta:
    def __init__(self, pixel_size_um_embedded, pixel_size_um_estimated, ):
        self.additional = {}
        self.pixel_size_estimated = pixel_size_um_estimated
        self.pixel_size_embedded = pixel_size_um_estimated
        self.adata_nb_col = adata_nb_col
        self.fullres_px_width = fullres_px_width
        self.fullres_px_height = fullres_px_height"""
    
    
@dataclass(frozen=True)
class HESTData:
    """
    Object representing a single Spatial Transcriptomics sample along with a full resolution H&E image and metadatas
    """
    h5_path = None
    spatial_path = None
    save_positions = True
    
    
    def _verify_format(self, adata):
        assert 'spatial' in adata.obsm
        for field in ['in_tissue', 'array_row', 'array_col']:
            if field not in adata.obs.columns:
                raise ValueError('{field} column missing in adata.obs')
        try:
            adata.uns['spatial']['ST']['images']['downscaled_fullres']
        except KeyError:
            raise ValueError('Downscaled image missing in adata.obs')
            
    
    
    def __init__(
        self, 
        adata: sc.AnnData,
        img: np.ndarray, 
        meta: Dict, 
        spot_size: float, 
        spot_inter_dist: float
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
                and the following collomns in adata.obs: ['in_tissue', 'array_row', 'array_col']
            img (np.ndarray): Full resolution image corresponding to the ST data
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
        """
        self.adata = adata
        
        self.img = img
        self.meta = meta
        self._verify_format(adata)
        self.pixel_size_embedded = meta['pixel_size_um_embedded']
        self.pixel_size_estimated = meta['pixel_size_um_estimated']
        self.spots_under_tissue = meta['spots_under_tissue']
        
        
    
    def __repr__(self):
        rep = f"""'pixel_size_um_embedded' is {self.pixel_size_embedded}
        'pixel_size_um_estimated' is {self.pixel_size_estimated}
        'spots_under_tissue' is {self.spots_under_tissue}"""
        return rep
        
    
    def save_spatial_plot(self, save_path: str):
        """Save the spatial plot from that STObject

        Args:
            save_path (str): _description_
        """
        print("Plotting spatial plots...")
             
        sc.pl.spatial(self.adata, show=None, img_key="downscaled_fullres", color=['total_counts'], title=f"in_tissue spots")
        
        filename = f"spatial_plots.png"
        
        # Save the figure
        plt.savefig(os.path.join(save_path, filename))
        plt.close()  # Close the plot to free memory
        print(f"H&E overlay spatial plots saved in {save_path}")
    
        
    def save(self, path: str):
        try:
            self.adata.write(os.path.join(path, 'aligned_adata.h5ad'))
        except:
            #traceback.print_exc()
            # workaround from https://github.com/theislab/scvelo/issues/255
            self.adata.__dict__['_raw'].__dict__['_var'] = self.adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
            self.adata.write(os.path.join(path, 'aligned_adata.h5ad'))
        
        if self.h5_path is not None:
            shutil.copy(self.h5_path, os.path.join(path, 'filtered_feature_bc_matrix.h5'))
        else:
            write_10X_h5(self.adata, os.path.join(path, 'filtered_feature_bc_matrix.h5'))
        
        if self.spatial_path is not None:
            shutil.copytree(self.spatial_path, os.path.join(path, 'spatial'), dirs_exist_ok=True)
        else:
            os.makedirs(os.path.join(path, 'spatial'), exist_ok=True)
            _save_scalefactors(self.adata, os.path.join(path, 'spatial/scalefactors_json.json'))

        df = self.adata.obs
        
        if self.save_positions:
            tissue_positions = df[['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']]
            tissue_positions.to_csv(os.path.join(path, 'spatial/tissue_positions.csv'), index=True, index_label='barcode')
        
        self.meta['adata_nb_col'] = len(self.adata.var_names)
        self.meta['fullres_px_width'] = self.img.shape[1]
        self.meta['fullres_px_height'] = self.img.shape[0]
        with open(os.path.join(path, 'metrics.json'), 'w') as json_file:
            json.dump(self.meta, json_file) 
        
        downscaled_img = self.adata.uns['spatial']['ST']['images']['downscaled_fullres']
        down_fact = self.adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
        down_img = Image.fromarray(downscaled_img)
        down_img.save(os.path.join(path, 'downscaled_fullres.jpeg'))
        
        pixel_size_embedded = self.meta['pixel_size_um_embedded']
        pixel_size_estimated = self.meta['pixel_size_um_estimated']
        
        
        _plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, os.path.join(path, 'pixel_size_vis.png'))
        use_embedded = not self.save_positions
        
        write_wsi(self.img, os.path.join(path, ALIGNED_HE_FILENAME), self.meta, use_embedded_size=use_embedded)
        
        
    def plot_genes(self, path, top_k=300, plot_spatial=True):
        #self.adata.obs['in_tissue_cat'] = self.adata.obs['in_tissue_cat'].astype('category')
        #sc.tl.rank_genes_groups(self.adata, groupby='in_tissue', method='wilcoxon')
        sums = np.array(np.sum(self.adata.X, axis=0))[0]

        # Sort genes based on variability
        top_genes_mask = np.argsort(-sums)[:top_k]  # Sort in descending order
        top_genes = self.adata.var_names[top_genes_mask]
        
        
        print('saving gene plots...')
        FIGSIZE = (15, 5)
        old_figsize = rcParams["figure.figsize"]
        os.makedirs(os.path.join(path, 'gene_plots'), exist_ok=True)
        if os.path.exists(os.path.join(path, 'gene_bar_plots')):
            # Remove the directory if it exists
            shutil.rmtree(os.path.join(path, 'gene_bar_plots'))
        os.makedirs(os.path.join(path, 'gene_bar_plots'), exist_ok=True)

        #gene_names = [name for name in self.adata.var_names if ('BLANK' not in name and 'NegControl' not in name)]
        gene_names = top_genes

        adata_df = self.adata.to_df()
        for gene_name in tqdm(gene_names):
            col = adata_df[gene_name]
            plt.close()
            if plot_spatial:
                sc.pl.spatial(self.adata, show=None, img_key="downscaled_fullres", color=gene_name) 
                plt.savefig(os.path.join(path, 'gene_plots', f'{gene_name}.png'))
                plt.close()  # Close the plot to free memory     
            else:
                rcParams["figure.figsize"] = FIGSIZE
                plt.hist(col.values, bins=50, range=(0, 2000))
                # Add labels and title
                plt.ylabel(f'{gene_name} count per spot')            
                plt.savefig(os.path.join(path, 'gene_bar_plots', f'{gene_name}.png'))
                plt.close()  # Close the plot to free memory
        rcParams["figure.figsize"] = old_figsize


class VisiumData(HESTData): 
    def __init__(self, adata: sc.AnnData, img: np.ndarray, meta: Dict):
        super().__init__(adata, img, meta, spot_size=55., spot_inter_dist=100.)
        
class STData(HESTData):
    def __init__(self, adata: sc.AnnData, img: np.ndarray, meta: Dict):
        super().__init__(adata, img, meta, spot_size=100., spot_inter_dist=200.)
        
class XeniumData(HESTData):
    def __init__(self, adata: sc.AnnData, img: np.ndarray, meta: Dict):
        super().__init__(adata, img, meta, spot_size=55., spot_inter_dist=100.)
        

class Reader:
    
    def auto_read(self, path: str) -> HESTData:
        """
        Automatically detect the file names and determine a reading strategy based on the
        detected files. For a more control on the reading process, consider using `read()` instead

        Args:
            path (st): path to the directory containing all the necessary files

        Returns:
            STObject: STObject that was read
        """
        
        hest_object = self._auto_read(path)

        os.makedirs(os.path.join(path, 'processed'), exist_ok=True)
        
        hest_object.adata.var["mito"] = hest_object.adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(hest_object.adata, qc_vars=["mito"], inplace=True)
        
        return hest_object
    
    @abstractmethod
    def _auto_read(self, path) -> HESTData:
        pass
  
    def read(self, **options) -> HESTData:
        raise NotImplementedError("Reader is an abstract class invoke this method from one of the subclass instead")


class VisiumReader(Reader):
    
    def _auto_read(self, path) -> HESTData:
        custom_adata = None
        img_filename = _find_biggest_img(path)
        
        # move files to right folders
        tissue_positions_path = _find_first_file_endswith(path, 'tissue_positions_list.csv')
        if tissue_positions_path is None:
            tissue_positions_path = _find_first_file_endswith(path, 'tissue_positions.csv')
        scalefactors_path = _find_first_file_endswith(path, 'scalefactors_json.json')
        hires_path = _find_first_file_endswith(path, 'tissue_hires_image.png')
        lowres_path = _find_first_file_endswith(path, 'tissue_lowres_image.png')
        spatial_coord_path = _find_first_file_endswith(path, 'spatial')
        raw_count_path = _find_first_file_endswith(path, 'raw_count.txt')
        if spatial_coord_path is None and (tissue_positions_path is not None or \
                scalefactors_path is not None or hires_path is not None or \
                lowres_path is not None or spatial_coord_path is not None):
            os.makedirs(os.path.join(path, 'spatial'), exist_ok=True)
            spatial_coord_path = _find_first_file_endswith(path, 'spatial')
        
        if tissue_positions_path is not None:
            shutil.move(tissue_positions_path, spatial_coord_path)
        if scalefactors_path is not None:
            shutil.move(scalefactors_path, spatial_coord_path)
        if hires_path is not None:
            shutil.move(hires_path, spatial_coord_path)
        if lowres_path is not None:
            shutil.move(lowres_path, spatial_coord_path)
        
            
        filtered_feature_path = _find_first_file_endswith(path, 'filtered_feature_bc_matrix.h5')
        raw_feature_path = _find_first_file_endswith(path, 'raw_feature_bc_matrix.h5')
        alignment_path = _find_first_file_endswith(path, 'alignment_file.json')
        if alignment_path is None:
            alignment_path = _find_first_file_endswith(path, 'alignment.json')
        if alignment_path is None:
            alignment_path = _find_first_file_endswith(path, 'alignment', anywhere=True)
        if alignment_path is None and os.path.exists(os.path.join(path, 'spatial')):
            alignment_path = _find_first_file_endswith(os.path.join(path, 'spatial'), 'autoalignment.json')
        if alignment_path is None:
            json_path = _find_first_file_endswith(path, '.json')
            if json_path is not None:
                f = open(json_path)
                meta = json.load(f)
                if 'oligo' in meta:
                    alignment_path = json_path
        mex_path = _find_first_file_endswith(path, 'mex')
        
        mtx_path = _find_first_file_endswith(path, 'matrix.mtx.gz')
        mtx_path = mtx_path if mtx_path is not None else  _find_first_file_endswith(path, 'matrix.mtx')
        features_path = _find_first_file_endswith(path, 'features.tsv.gz')
        features_path = features_path if features_path is not None else  _find_first_file_endswith(path, 'features.tsv')
        barcodes_path = _find_first_file_endswith(path, 'barcodes.tsv.gz')
        barcodes_path = barcodes_path if barcodes_path is not None else  _find_first_file_endswith(path, 'barcodes.tsv')
        if mex_path is None and (mtx_path is not None or features_path is not None or barcodes_path is not None):
            os.makedirs(os.path.join(path, 'mex'), exist_ok=True)
            mex_path = _find_first_file_endswith(path, 'mex')
            shutil.move(mtx_path, mex_path)
            shutil.move(features_path, mex_path)
            shutil.move(barcodes_path, mex_path)
        
        # TODO remove
        GSE234047_count_path = _find_first_file_endswith(path, '_counts.csv')
        GSE180128_count_path = None
        if "Comprehensive Atlas of the Mouse Urinary Bladder" in path:
            GSE180128_count_path = _find_first_file_endswith(path, '.csv')
            
        GSE167096_count_path = None
        if "Spatial Transcriptomics of human fetal liver"  in path:
            custom_adata = GSE167096_to_adata(path)
            #GSE167096_count_path = _find_first_file_endswith(path, 'symbol.txt')
            
        GSE203165_count_path = None
        if 'Spatial sequencing of Foreign body granuloma' in path:
            GSE203165_count_path = _find_first_file_endswith(path, 'raw_counts.txt')
            
            
        use_adata_align = False
            
        if 'YAP Drives Assembly of a Spatially Colocalized Cellular Triad Required for Heart Renewal' in path:
            custom_adata = GSE217828_to_custom(path)
            use_adata_align = True
            
        if 'Spatiotemporal mapping of immune and stem cell dysregulation after volumetric muscle loss' in path:
            my_path = _find_first_file_endswith(path, '.h5ad')
            custom_adata = sc.read_h5ad(my_path)
            
        if 'The neurons that restore walking after paralysis [spatial transcriptomics]' in path:
            my_path = _find_first_file_endswith(path, '.h5ad')
            custom_adata = sc.read_h5ad(my_path)           
            
        seurat_h5_path = _find_first_file_endswith(path, 'seurat.h5ad')
        
        if img_filename is None:
            raise Exception(f"Couldn't detect an image in the directory {path}")
        
        metric_file_path = _find_first_file_endswith(path, 'metrics_summary.csv')
        
        st_object = self.read(
            filtered_bc_matrix_path=filtered_feature_path,
            raw_bc_matrix_path=raw_feature_path,
            spatial_coord_path=spatial_coord_path,
            img_path=os.path.join(path, img_filename),
            alignment_file_path=alignment_path,
            mex_path=mex_path,
            raw_count_path=raw_count_path,
            GSE234047_count_path=GSE234047_count_path,
            GSE180128_count_path=GSE180128_count_path,
            GSE203165_count_path=GSE203165_count_path,
            seurat_h5_path=seurat_h5_path,
            metric_file_path=metric_file_path,
            custom_adata=custom_adata,
            use_adata_align=use_adata_align
        )
        
        os.makedirs(os.path.join(path, 'processed'), exist_ok=True)
        
        st_object.adata.var["mito"] = st_object.adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(st_object.adata, qc_vars=["mito"], inplace=True)
        
        st_object.h5_path = filtered_feature_path
        st_object.spatial_path = spatial_coord_path
        
        return st_object        
    
    
    def read(self,
        img_path: str,
        filtered_bc_matrix_path: str = None,
        raw_bc_matrix_path: str = None,
        spatial_coord_path: str = None,
        alignment_file_path: str = None, 
        mex_path: str = None,
        custom_matrix_path: str = None,
        raw_count_path: str = None,
        GSE234047_count_path: str = None,
        GSE180128_count_path: str = None,
        GSE203165_count_path: str = None,
        seurat_h5_path: str = None,
        metric_file_path: str = None,
        meta_dict: dict = {},
        custom_adata: sc.AnnData = None,
        use_adata_align: bool = False,
        force_auto_alignment: bool = False
    ) -> VisiumData:
        raw_bc_matrix = None
        
        print('alignment file is ', alignment_file_path)

        if filtered_bc_matrix_path is not None:
            adata = sc.read_10x_h5(filtered_bc_matrix_path)
        elif mex_path is not None:
            _helper_mex(mex_path, 'barcodes.tsv.gz')
            _helper_mex(mex_path, 'features.tsv.gz')
            _helper_mex(mex_path, 'matrix.mtx.gz')
                
            adata = sc.read_10x_mtx(mex_path)
        elif custom_adata is not None:
            adata = custom_adata
        elif raw_bc_matrix_path is not None:
            adata = sc.read_10x_h5(raw_bc_matrix_path)
        elif custom_matrix_path is not None:
            adata = _txt_matrix_to_adata(custom_matrix_path)
        elif raw_count_path is not None:
            adata = _raw_count_to_adata(raw_count_path)
        elif GSE234047_count_path is not None:
            adata = GSE234047_to_h5(GSE234047_count_path)
        elif GSE180128_count_path is not None:
            adata = GSE180128_to_h5(GSE180128_count_path)
        elif seurat_h5_path is not None:
            adata = sc.read_h5ad(seurat_h5_path)
        elif GSE203165_count_path is not None:
            adata = GSE203165_to_adata(GSE203165_count_path)
        else:
            raise Exception(f"Couldn't find gene expressions, make sure to provide at least a filtered_bc_matrix.h5 or a mex folder")

        adata.var_names_make_unique()
        print(adata)

        img, pixel_size_embedded = _load_image(img_path)
        
        
        print('trim the barcodes')
        adata.obs.index = [idx[:18] for idx in adata.obs.index]
        if not adata.obs.index[0].endswith('-1'):
            print("barcode don't end with -1 !")
            adata.obs.index = [idx + '-1' for idx in adata.obs.index]

            
        if not use_adata_align:
            tissue_positions_path = _find_first_file_endswith(spatial_coord_path, 'tissue_positions.csv', exclude='aligned_tissue_positions.csv')
            tissue_position_list_path = _find_first_file_endswith(spatial_coord_path, 'tissue_positions_list.csv')
            if tissue_positions_path is not None or tissue_position_list_path is not None and not force_auto_alignment:
                #tissue_positions_path = _find_first_file_endswith(spatial_coord_path, 'tissue_positions.csv')
                if tissue_positions_path is not None:
                    tissue_positions = pd.read_csv(tissue_positions_path, sep=",", na_filter=False, index_col=0) 
                else:
                    tissue_positions_path = _find_first_file_endswith(spatial_coord_path, 'tissue_positions_list.csv')
                    tissue_positions = pd.read_csv(tissue_positions_path, header=None, sep=",", na_filter=False, index_col=0)
                    
                    tissue_positions = tissue_positions.rename(columns={1: "in_tissue", # in_tissue: 1 if spot is captured in tissue region, 0 otherwise
                                                    2: "array_row", # spot row index
                                                    3: "array_col", # spot column index
                                                    4: "pxl_row_in_fullres", # spot x coordinate in image pixel
                                                    5: "pxl_col_in_fullres"}) # spot y coordinate in image pixel

                tissue_positions.index = [idx[:18] for idx in tissue_positions.index]
                spatial_aligned = _align_tissue_positions(
                    alignment_file_path, 
                    tissue_positions, 
                    adata
                )

                assert np.array_equal(spatial_aligned.index, adata.obs.index)

            elif alignment_file_path is not None and not force_auto_alignment:
                spatial_aligned = _alignment_file_to_tissue_positions(alignment_file_path, adata)
            else:
                print('no tissue_positions_list.csv/tissue_positions.csv or alignment file found')
                print('attempt fiducial auto alignment...')

                os.makedirs(os.path.join(os.path.dirname(img_path), 'spatial'), exist_ok=True)
                autoalign_with_fiducials(img, os.path.join(os.path.dirname(img_path), 'spatial'))
                
                autoalignment_file_path = os.path.join(os.path.dirname(img_path), 'spatial', 'autoalignment.json')
                spatial_aligned = _alignment_file_to_tissue_positions(autoalignment_file_path, adata)

            
            col1 = spatial_aligned['pxl_col_in_fullres'].values
            col2 = spatial_aligned['pxl_row_in_fullres'].values
            
            matrix = np.vstack((col1, col2)).T
            
            adata.obsm['spatial'] = matrix
        else:
            spatial_aligned = adata.obs[['in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]
        
        scalefactors_path = _find_first_file_endswith(spatial_coord_path, 'scalefactors_json.json')

        pixel_size, spot_estimate_dist = _find_pixel_size_from_spot_coords(spatial_aligned)
        

        adata.obs = spatial_aligned
            
        downscaled_img, down_fact = _register_downscale_img(adata, img, pixel_size)
        
        dict = {}
        if metric_file_path is not None:
            dict = _metric_file_do_dict(metric_file_path)
            
        dict['pixel_size_um_embedded'] = pixel_size_embedded
        dict['pixel_size_um_estimated'] = pixel_size
        dict['fullres_height'] = img.shape[0]
        dict['fullres_width'] = img.shape[1]
        dict['spots_under_tissue'] = len(adata.obs)
        dict['spot_estimate_dist'] = int(spot_estimate_dist)
        dict['spot_diameter'] = 55.
        dict['inter_spot_dist'] = 100.
        
        dict = {**meta_dict, **dict}
        

        return VisiumData(adata, img, dict)
    
    
class STReader(Reader):
    
    def _auto_read(self, path) -> STData:
        packing = SpotPacking.GRID_PACKING
        meta_table_path = None
        custom_adata = None
        inter_spot_dist = 200
        spot_diameter = 100

        img_path = _find_biggest_img(path)
        
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
            custom_adata = self._ADT_to_adata(os.path.join(path, img_path), raw_counts_path)
            
        if "Single Cell and Spatial Analysis of Human Squamous Cell Carcinoma [ST]" in path:
            custom_adata = self._GSE144239_to_adata(raw_counts_path, spot_coord_path)
            packing = SpotPacking.GRID_PACKING
            inter_spot_dist = 110.
            spot_diameter = 150.

        #if 'Spatial Transcriptomics of human fetal liver' in path:
        #    GSE167096_count_path = _find_first_file_endswith(path, 'table_symbol.txt')
        #    custom_adata = GSE167096_to_h5(GSE167096_count_path)
            
            
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
            custom_adata=custom_adata)
        
        sc.pp.calculate_qc_metrics(st_object.adata, inplace=True)
        
        os.makedirs(os.path.join(path, 'processed'), exist_ok=True) 
        
        st_object.save_positions = False
        
        return st_object
    
    
    def _GSE144239_to_adata(self, raw_counts_path, spot_coord_path):
        raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
        spot_coord = pd.read_csv(spot_coord_path, sep='\t')
        spot_coord.index = spot_coord['x'].astype(str) + ['x' for _ in range(len(spot_coord))] + spot_coord['y'].astype(str)
        merged = pd.merge(spot_coord, raw_counts, left_index=True, right_index=True)
        raw_counts = raw_counts.reindex(merged.index)
        adata = sc.AnnData(raw_counts)
        col1 = merged['pixel_x'].values
        col2 = merged['pixel_y'].values
        matrix = (np.vstack((col1, col2))).T
        adata.obsm['spatial'] = matrix
        return adata
    
    
    def _ADT_to_adata(self, img_path, raw_counts_path):
        basedir = os.path.dirname(img_path)
        # combine spot coordinates into a single dataframe
        pre_adt_path= _find_first_file_endswith(basedir, 'pre-ADT.tsv')
        post_adt_path = _find_first_file_endswith(basedir, 'postADT.tsv')
        if post_adt_path is None:
            post_adt_path = _find_first_file_endswith(basedir, 'post-ADT.tsv')
        counts = pd.read_csv(raw_counts_path, index_col=0, sep='\t')
        pre_adt = pd.read_csv(pre_adt_path, sep='\t')
        post_adt = pd.read_csv(post_adt_path, sep='\t')
        merged_coords = pd.concat([pre_adt, post_adt], ignore_index=True)
        merged_coords.index = [str(x) + 'x' + str(y) for x, y in zip(merged_coords['x'], merged_coords['y'])]
        merged = pd.merge(merged_coords, counts, left_index=True, right_index=True, how='inner')
        counts = counts.reindex(merged.index)
        adata = sc.AnnData(counts)
        col1 = merged['pixel_x'].values
        col2 = merged['pixel_y'].values
        matrix = (np.vstack((col1, col2))).T
        adata.obsm['spatial'] = matrix
        return adata
    
    
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
    ) -> STData:
        #raw_counts = pd.read_csv(raw_counts_path, sep='\t')
        img, pixel_size_embedded = _load_image(img_path)
        
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
        
        # TODO might not be precise if we use round on the column and row
        pixel_size, spot_estimate_dist = _find_pixel_size_from_spot_coords(my_df, inter_spot_dist=inter_spot_dist, packing=packing)
        _register_downscale_img(adata, img, pixel_size, spot_size=spot_diameter)
        
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
        
        return STData(adata, img, dict)
    
    
class XeniumReader(Reader):
    def _auto_read(self, path) -> XeniumData:
        img_filename = _find_biggest_img(path)
                
        alignment_path = None
        for file in os.listdir(path):
            if file.endswith('imagealignment.csv'):
                alignment_path = os.path.join(path, file)
        
        st_object = self.read(
            path=path,
            feature_matrix_path=os.path.join(path, 'cell_feature_matrix.h5'), 
            transcripts_path=os.path.join(path, 'transcripts.parquet'), 
            img_path=os.path.join(path, img_filename), 
            alignment_file_path=alignment_path
        )
        
        os.makedirs(os.path.join(path, 'processed'), exist_ok=True)
        
        sc.pp.calculate_qc_metrics(st_object.adata, inplace=True)
        
        #write_wsi(st_object.img, os.path.join(path, 'processed', ALIGNED_HE_FILENAME), self.meta, use_embedded_size=True)
        
        return st_object
    
    
    def __xenium_estimate_pixel_size(self, pixel_size_morph, he_to_morph_matrix):
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
         
    
    
    def read(
        self,
        path: str, 
        feature_matrix_path: str, 
        transcripts_path: str,
        img_path: str,
        alignment_file_path: str,
        plot_genes: bool = False,
        use_cache: bool = False
    ) -> XeniumData:
        basedir = os.path.dirname(img_path)
        
        experiment_path = _find_first_file_endswith(path, 'experiment.xenium')
        with open(experiment_path) as f:
            pixel_size_embedded = json.load(f)['pixel_size']
        
        
        dict = {}
        dict['pixel_size_um_embedded'] = pixel_size_embedded
        dict['pixel_size_um_estimated'] = None
        
        cur_dir = os.path.dirname(transcripts_path)   
        
        adata = sc.read_10x_h5(
            filename=feature_matrix_path
        )
        
        img, pixel_size_embedded = _load_image(img_path)
        
        experiment_file = open(os.path.join(cur_dir, 'experiment.xenium'))
        dict_exp = json.load(experiment_file)

        pixel_size_morph = dict_exp['pixel_size']
        dict['spot_diameter'] = 55.
        dict['inter_spot_dist'] = 100.
        
        dict = {**dict, **dict_exp}
        
        if os.path.exists(os.path.join(cur_dir, 'cached_pseudo_visium.h5ad')) and use_cache:
            adata = sc.read_h5ad(os.path.join(cur_dir, 'cached_pseudo_visium.h5ad'))
            cached_metrics = json.load(open(os.path.join(cur_dir, 'cached_metrics.json')))
            dict['pixel_size_um_embedded'] = cached_metrics['pixel_size_um_embedded']
            dict['pixel_size_um_estimated'] = cached_metrics['pixel_size_um_estimated']
        else:
            
            df_transcripts = pd.read_parquet(transcripts_path)
            
            if alignment_file_path is not None:
                print('found an alignment file, aligning transcripts...')
                alignment_file = pd.read_csv(alignment_file_path, header=None)
                alignment_matrix = alignment_file.values
                #convert alignment matrix from pixel to um
                alignment_matrix[0][2] *= pixel_size_morph
                alignment_matrix[1][2] *= pixel_size_morph
                he_to_morph_matrix = alignment_matrix
                alignment_matrix = np.linalg.inv(alignment_matrix)
                coords = np.column_stack((df_transcripts["x_location"].values, df_transcripts["y_location"].values, np.ones((len(df_transcripts),))))
                aligned = (alignment_matrix @ coords.T).T
                df_transcripts['y_location'] = aligned[:,1]
                df_transcripts['x_location'] = aligned[:,0]
                pixel_size_estimated = self.__xenium_estimate_pixel_size(pixel_size_morph, he_to_morph_matrix)
                dict['pixel_size_um_estimated'] = pixel_size_estimated
            else:
                dict['pixel_size_um_estimated'] = pixel_size_morph
                
            
            # convert transcripts position from um to pixel
            df_transcripts["x_location"] = df_transcripts["x_location"] / pixel_size_morph
            df_transcripts["y_location"] = df_transcripts["y_location"] / pixel_size_morph
            
            
            
            
            adata = xenium_to_pseudo_visium(df_transcripts, dict['pixel_size_um_estimated'])
        
            _register_downscale_img(adata, img, dict['pixel_size_um_estimated'])
            
            adata.write_h5ad(os.path.join(cur_dir, 'cached_pseudo_visium.h5ad'))
            with open(os.path.join(cur_dir, 'cached_metrics.json'), 'w') as f:
                json.dump(dict, f)
            
        dict['spots_under_tissue'] = len(adata.obs)
        
        if plot_genes:
            print('saving gene plots...')
            FIGSIZE = (15, 5)
            old_figsize = rcParams["figure.figsize"]
            rcParams["figure.figsize"] = FIGSIZE
            os.makedirs(os.path.join(cur_dir, 'gene_plots'), exist_ok=True)
            os.makedirs(os.path.join(cur_dir, 'gene_bar_plots'), exist_ok=True)

            gene_names = [name for name in adata.var_names if ('BLANK' not in name and 'NegControl' not in name)]

            adata_df = adata.to_df()
            for gene_name in tqdm(gene_names):
                col = adata_df[gene_name]
                plt.close()
                #sc.pl.spatial(adata, show=None, img_key="downscaled_fullres", color=gene_name)
                plt.hist(col.values, bins=50, range=(0, 2000))

                # Add labels and title
                plt.ylabel(f'{gene_name} count per spot')
                
                plt.savefig(os.path.join(cur_dir, 'gene_bar_plots', f'{gene_name}.png'))
                plt.close()  # Close the plot to free memory
            rcParams["figure.figsize"] = old_figsize
            

        st_object =  XeniumData(adata, img, dict)
        st_object.save_positions = False
        return st_object
    

def reader_factory(path: str) -> Reader:
    path = path.lower()
    if 'visium-hd' in path:
        raise NotImplementedError('implement visium-hd')
    elif 'visium' in path:
        return VisiumReader()
    elif 'xenium' in path:
        return XeniumReader()
    elif 'st' in path:
        return STReader()
        
    
def read_and_save(path, save_plots=True, plot_genes=False):
    print(f'Reading from {path}...')
    reader = reader_factory(path)
    st_object = reader.auto_read(path)
    print(st_object)
    save_path = os.path.join(path, 'processed')
    st_object.save(save_path)
    if save_plots:
        st_object.save_spatial_plot(save_path)
    if plot_genes:
        st_object.plot_genes(save_path, top_k=300)
        
        
def process_meta_df(meta_df, save_spatial_plots=True, plot_genes=False):
    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        path = _get_path_from_meta_row(row)
        adata = read_and_save(path, save_plots=save_spatial_plots, plot_genes=plot_genes)