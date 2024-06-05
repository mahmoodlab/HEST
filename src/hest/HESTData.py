import json
import os
import shutil
from functools import partial
from typing import Dict, Tuple, Union
import warnings

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile

try:
    import openslide
except Exception:
    print("Couldn't import openslide, verify that openslide is installed on your system, https://openslide.org/download/")
import pandas as pd

try:
    import pyvips
except Exception:
    print("Couldn't import pyvips, verify that libvips is installed on your system")
import dask.array as da
import scanpy as sc
from dask import delayed
from dask.array import from_delayed
from matplotlib import rcParams
from matplotlib.collections import PatchCollection
from PIL import Image
from spatial_image import SpatialImage
from spatialdata import SpatialData
from tqdm import tqdm

from .masking import (apply_otsu_thresholding, keep_largest_area,
                      mask_to_contours, save_pkl, scale_contour_dim)
from .utils import (ALIGNED_HE_FILENAME, get_path_from_meta_row, load_image,
                    plot_verify_pixel_size, tiff_save)
from .vst_save_utils import initsave_hdf5


class HESTData:
    """
    Object representing a Spatial Transcriptomics sample along with a full resolution H&E image and metadatas
    """
    
    tissue_mask: np.ndarray = None
    """tissue mask for that sample, will be None until _compute_mask() is called"""
    
    contours_tissue: list = None
    """tissue contours for that sample, will be None until _compute_mask() is called"""
    
    cellvit_seg = None
    """dictionary of cells in the CellViT .geojson format"""
    
    img = None
    """WSI image associated with this sample, will be None until loaded"""
    
    
    def _verify_format(self, adata):
        assert 'spatial' in adata.obsm
        try:
            adata.uns['spatial']['ST']['images']['downscaled_fullres']
        except KeyError:
            raise ValueError('Downscaled image missing in adata.obs')
        
        features = adata.obs.columns
        required_features = ['array_col', 'array_row', 'in_tissue', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
        missing = []
        for req in required_features:
            if not req in features:
                missing.append(req)
        if len(missing) > 0:
            raise ValueError(f'The following columns are missing in adata.obs: {missing}')
        
    
    def __init__(
        self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None
    ):
        """
        class representing a single ST profile + its associated WSI image
        
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
                and the following collomns in adata.obs: ['array_col', 'array_row', 'in_tissue', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
        """
        self.adata = adata
        
        if isinstance(img, str):
            self.wsi = openslide.OpenSlide(img)
            self.img = None
        else:
            self.img = img
            
        self.meta = meta
        self._verify_format(adata)
        self.pixel_size = pixel_size
        self.cellvit_seg = cellvit_seg
        
        if 'total_counts' not in self.adata.var_names:
            sc.pp.calculate_qc_metrics(self.adata, inplace=True)
        
        
    def __repr__(self):
        sup_rep = super().__repr__()

        img_str = 'WSI in memory'if self.is_image_in_mem() else "WSI not in memory"

        height, width = self.get_img_dim()
        dim_str = f'WSI has dim height={height}, width={width}'
    
        rep = f"""{sup_rep}
        'pixel_size' is {self.pixel_size}
        {img_str}
        {dim_str}
        """
        return rep
        
    
    def save_spatial_plot(self, save_path: str, pl_kwargs={}):
        """Save the spatial plot from that STObject

        Args:
            save_path (str): path to a directory where the spatial plot will be saved
            pl_kwargs(Dict): arguments for sc.pl.spatial
        """
        print("Plotting spatial plots...")
             
        sc.pl.spatial(self.adata, show=None, img_key="downscaled_fullres", color=['total_counts'], title=f"in_tissue spots", **pl_kwargs)
        
        filename = f"spatial_plots.png"
        
        # Save the figure
        plt.savefig(os.path.join(save_path, filename))
        plt.close()  # Close the plot to free memory
        print(f"H&E overlay spatial plots saved in {save_path}")
    
    
    def _load_wsi(self):
        self.img, _ = load_image(self.wsi._filename)
        
    
    def get_img(self):
        if self.img is None:
            self._load_wsi()
        return self.img
    
    
    def get_img_dim(self) -> Tuple[int, int]:
        """ get the fullres image dimension as (height, width)

        Returns:
            Tuple[int, int]: (height, width)
        """

        if self.img is not None:
            shape = self.img.shape[:2]
            return shape[0], shape[1]
        else:
            width, height = self.wsi.dimensions
            return height, width
    
        
    def save(self, path: str, pyramidal=True, bigtiff=False, plot_pxl_size=False):
        """Save a HESTData object to `path` as follows:
            - aligned_adata.h5ad (contains expressions for each spots + their location on the fullres image + a downscaled version of the fullres image)
            - metrics.json (contains useful metrics)
            - downscaled_fullres.jpeg (a downscaled version of the fullres image)
            - aligned_fullres_HE.tif (the full resolution image)
            - cells.geojson (cell segmentation if it exists)

        Args:
            path (str): save location
            pyramidal (bool, optional): whenever to save the full resolution image as pyramidal (can be slow to save, however it's sometimes necessary for loading large images in QuPath). Defaults to True.
            bigtiff (bool, optional): whenever the bigtiff image is more than 4.1GB. Defaults to False.
        """
        try:
            self.adata.write(os.path.join(path, 'aligned_adata.h5ad'))
        except:
            # workaround from https://github.com/theislab/scvelo/issues/255
            self.adata.__dict__['_raw'].__dict__['_var'] = self.adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
            self.adata.write(os.path.join(path, 'aligned_adata.h5ad'))
        
        
        img = self.get_img()
        self.meta['adata_nb_col'] = len(self.adata.var_names)
        self.meta['fullres_px_width'] = img.shape[1]
        self.meta['fullres_px_height'] = img.shape[0]
        with open(os.path.join(path, 'metrics.json'), 'w') as json_file:
            json.dump(self.meta, json_file) 
            
        with open(os.path.join(path, 'cells.geojson')) as json_file:
            json.dump(self.cellvit_seg, json_file)
        
        downscaled_img = self.adata.uns['spatial']['ST']['images']['downscaled_fullres']
        down_fact = self.adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
        down_img = Image.fromarray(downscaled_img)
        down_img.save(os.path.join(path, 'downscaled_fullres.jpeg'))
        
        
        if plot_pxl_size:
            pixel_size_embedded = self.meta['pixel_size_um_embedded']
            pixel_size_estimated = self.meta['pixel_size_um_estimated']
            
            
            plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, os.path.join(path, 'pixel_size_vis.png'))

        
        tiff_save(img, os.path.join(path, ALIGNED_HE_FILENAME), self.pixel_size, pyramidal=pyramidal, bigtiff=bigtiff)
        
        
    def plot_genes(self, path, top_k=300, plot_spatial=True):
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
        
    
    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        """Get a downscaled version of the full resolution image

        Args:
            width (int): width of thumbnail
            height (int): height of thumbnail

        Returns:
            np.ndarray: thumbnail
        """
        if self.img is not None:
            thumb = np.array(cv2.resize(self.img, dsize=(width, height)))
        else:
            thumb = np.array(self.wsi.get_thumbnail((width, height)))
        return thumb      
        

    def visualize_mask_and_patches(
                self,
                line_color=(0, 255, 0),
                hole_color=(0, 0, 255),
                line_thickness=250,
                target_width=1000,
                view_slide_only=False,
                seg_display=True,
                tissue_mask=None
        ):
            height, width = self.get_img_dim()
            downsample = target_width / width

            top_left = (0,0)
            scale = [downsample, downsample]    

            img = self.get_thumbnail(round(width * downsample), round(height * downsample))

            self.downscaled_img = img.copy()

            if tissue_mask is None:
                tissue_mask = self.get_tissue_mask()

            downscaled_mask = cv2.resize(tissue_mask, (img.shape[1], img.shape[0]))
            downscaled_mask = np.expand_dims(downscaled_mask, axis=-1)
            downscaled_mask = downscaled_mask * np.array([0, 0, 0]).astype(np.uint8)

            if view_slide_only:
                return Image.fromarray(img)

            offset = tuple(-(np.array(top_left) * scale).astype(int))
            draw_cont = partial(cv2.drawContours, contourIdx=-1, thickness=line_thickness, lineType=cv2.LINE_8, offset=offset)
            draw_cont_fill = partial(cv2.drawContours, contourIdx=-1, thickness=cv2.FILLED, offset=offset)

            if self.contours_tissue is not None and seg_display:
                for _, cont in enumerate(self.contours_tissue):
                    cont = np.array(scale_contour_dim(cont, scale))
                    draw_cont(image=img, contours=[cont], color=line_color)
                    draw_cont_fill(image=downscaled_mask, contours=[cont], color=line_color)

                ### Draw hole contours
                for cont in self.contours_holes:
                    cont = scale_contour_dim(cont, scale)
                    draw_cont(image=img, contours=cont, color=hole_color) 

            alpha = 0.4
            self.downscaled_mask = downscaled_mask
            self.tissue_mask = cv2.resize(downscaled_mask, self.tissue_mask.shape).round().astype(np.uint8)
            img = cv2.addWeighted(img, 1 - alpha, downscaled_mask, alpha, 0)
            img = img.astype(np.uint8)

            return Image.fromarray(img)


    def _compute_mask(self, keep_largest=False, thumbnail_width=2000):
        height, width = self.get_img_dim()
        scale = thumbnail_width / width
        thumbnail = self.get_thumbnail(round(width * scale), round(height * scale))
        mask = apply_otsu_thresholding(thumbnail).astype(np.uint8)
        mask = 1 - mask
        if keep_largest:
            mask = keep_largest_area(mask)
        self.tissue_mask = np.round(cv2.resize(mask, (width, height))).astype(np.uint8)
        self.contours_tissue, self.contours_holes = mask_to_contours(self.tissue_mask)


    def get_tissue_mask(self, keep_largest=False) -> np.ndarray:
        if self.tissue_mask is None:
            self._compute_mask(keep_largest)
        return self.tissue_mask
    

    def dump_patches(
        self,
        patch_save_dir: str,
        name: str = None,
        target_patch_size: int=224,
        target_pixel_size: float=0.5,
        verbose=0,
        dump_visualization=True,
        use_mask=True,
        load_in_memory=True,
        keep_largest=False
    ):
        
        adata = self.adata.copy()
        
        for index in adata.obs.index:
            if len(index) != len(adata.obs.index[0]):
                warnings.warn("indices of adata.obs should all have the same length to avoid problems when saving to h5", UserWarning)
                
        
        src_pixel_size =  self.pixel_size
        
        # minimum intersection percecentage with the tissue mask to keep a patch
        TISSUE_INTER_THRESH = 0.05
        TARGET_VIS_SIZE = 1000
        
        scale_factor = target_pixel_size / src_pixel_size
        patch_size_pxl = round(target_patch_size * scale_factor)
        patch_count = 0
        output_datafile = os.path.join(patch_save_dir, name + '.h5')

        assert len(adata.obs) == len(adata.obsm['spatial'])

        fig, ax = plt.subplots()
        
        mode_HE = 'w'
        i = 0
        img_height, img_width = self.get_img_dim()
        patch_rectangles = [] # lower corner (x, y) + (widht, height)
        downscale_vis = TARGET_VIS_SIZE / img_width

        if use_mask:
            tissue_mask = self.get_tissue_mask(keep_largest)
        else:
            tissue_mask = np.ones((img_height, img_width)).astype(np.uint8)

        mask_plot = self.visualize_mask_and_patches(line_thickness=3, target_width=1000, tissue_mask=tissue_mask)

        ax.imshow(mask_plot)
        for _, row in tqdm(adata.obs.iterrows(), total=len(adata.obs)):
            
            barcode_spot = row.name

            xImage = int(adata.obsm['spatial'][i][0])
            yImage = int(adata.obsm['spatial'][i][1])

            i += 1
            
            if not(0 <= xImage and xImage < img_width and 0 <= yImage and yImage < img_height):
                if verbose:
                    print('Warning, spot is out of the image, skipping')
                continue
            
            if not(0 <= yImage - patch_size_pxl // 2 and yImage + patch_size_pxl // 2 < img_height and \
                0 <= xImage - patch_size_pxl // 2 and xImage + patch_size_pxl // 2 < img_width):
                if verbose:
                    print('Warning, patch is out of the image, skipping')
                continue
            
            if load_in_memory:
                image_patch = self.img[yImage - patch_size_pxl // 2: yImage + patch_size_pxl // 2,
                                    xImage - patch_size_pxl // 2: xImage + patch_size_pxl // 2, :]
            else:
                image_patch = self.wsi.read_region((xImage - patch_size_pxl // 2, yImage - patch_size_pxl // 2), 0, (patch_size_pxl, patch_size_pxl))
            rect_x = (xImage - patch_size_pxl // 2) * downscale_vis
            rect_y = (yImage - patch_size_pxl // 2) * downscale_vis
            rect_width = patch_size_pxl * downscale_vis
            rect_height = patch_size_pxl * downscale_vis

            image_patch = np.array(image_patch)
            if image_patch.shape[2] == 4:
                image_patch = image_patch[:, :, :3]
                
            
            if use_mask:
                patch_mask = tissue_mask[yImage - patch_size_pxl // 2: yImage + patch_size_pxl // 2,
                                xImage - patch_size_pxl // 2: xImage + patch_size_pxl // 2]
                patch_area = patch_mask.shape[0] ** 2
                pixel_count = patch_mask.sum()

                if pixel_count / patch_area < TISSUE_INTER_THRESH:
                    continue

            patch_rectangles.append(matplotlib.patches.Rectangle((rect_x, rect_y), rect_width, rect_height))
            
            patch_count += 1
            image_patch = cv2.resize(image_patch, (target_patch_size, target_patch_size), interpolation=cv2.INTER_CUBIC)
            
            
            # Save ref patches
            assert image_patch.shape == (224, 224, 3)
            asset_dict = { 'img': np.expand_dims(image_patch, axis=0),  # (1 x w x h x 3)
                            'coords': np.expand_dims([yImage, xImage], axis=0),   # (1 x 2)
                            'barcode': np.expand_dims([barcode_spot], axis=0)
                            }

            attr_dict = {}
            attr_dict['img'] = {'patch_size': patch_size_pxl,
                                'factor': scale_factor}

            initsave_hdf5(output_datafile, asset_dict, attr_dict, mode=mode_HE, verbose=1)
            mode_HE = 'a'

        
        if dump_visualization:
            ax.add_collection(PatchCollection(patch_rectangles, facecolor='none', edgecolor='black', linewidth=0.3))
            ax.set_axis_off()
            os.makedirs(os.path.join(patch_save_dir, 'vis'), exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(patch_save_dir, 'vis', name + '.png'), dpi=400, bbox_inches = 'tight')
            
        if verbose:
            print(f'found {patch_count} valid patches')
            

    def get_tissue_contours(self) -> Dict[str, list]:
        """Get the tissue contours and holes

        Returns:
            Dict[str, list]: dictionnary of contours and holes in the tissue
        """
        if self.tissue_mask is None:
            self._compute_mask()
        
        asset_dict = {'holes': self.contours_holes, 
                      'tissue': self.contours_tissue, 
                      'groups': None}
        return asset_dict
    

    def save_tissue_seg_jpg(self, save_dir: str, name: str = 'hest') -> None:
        """Save tissue segmentation as a greyscale .jpg file

        Args:
            save_dir (str): path to save directory
            name (str): .jpg file is saved as {name}_mask.jpg
        """
        
        tissue_mask = self.get_tissue_mask()
        img = Image.fromarray(tissue_mask * 255)
        img.save(os.path.join(save_dir, f'{name}_mask.jpg'))
        
            
    def save_tissue_seg_pkl(self, save_dir: str, name: str) -> None:
        """Save tissue segmentation contour as a .pkl file

        Args:
            save_dir (str): path to pkl file
            name (str): .pkl file is saved as {name}_mask.pkl
        """

        asset_dict = self.get_tissue_contours()
        save_pkl(os.path.join(save_dir, f'{name}_mask.pkl'), asset_dict)


    def to_spatial_data(self, lazy_img=True) -> SpatialData:
        """Convert a HESTData sample to a scverse SpatialData object
        
        Args:
            lazy_img (bool, optional): whenever to lazily load the image if not already loaded (i.e. if self.img is None). Defaults to True.

        Returns:
            SpatialData: scverse SpatialData object
        """
        
        def read_hest_wsi(path):
            return pyvips.Image.tiffload(path).numpy().transpose((2, 0, 1))
    
        if lazy_img and self.img is None:
            if self.wsi is None:
                raise ValueError('HESTData.wsi path to a wsi needs to be set to use the lazy_img feature of to_spatial_data')
            
            with tifffile.TiffFile(self.wsi) as tif:
                page = tif.pages[0]
                width = page.imagewidth
                height = page.imagelength
                
            
            img = from_delayed(delayed(read_hest_wsi)(self.wsi), shape=(height, width, 3), dtype=np.int8)
        else:
            img = self.get_img()
            arr = da.from_array(img)
            sp_img = SpatialImage(arr, dims=['c', 'y', 'x'], attrs={'transform': None})
        
        st = SpatialData({'fullres': sp_img}, table=self.adata)
        
        #TODO add CellViT
        
        return st


class VisiumHESTData(HESTData): 
    def __init__(self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None
    ):
        super().__init__(adata, img, pixel_size, meta, cellvit_seg)

class VisiumHDHESTData(HESTData): 
    def __init__(self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
                and the following collomns in adata.obs: ['array_col', 'array_row', 'in_tissue', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
        """
        super().__init__(adata, img, pixel_size, meta, cellvit_seg)        
        
class STHESTData(HESTData):
    def __init__(self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
                and the following collomns in adata.obs: ['array_col', 'array_row', 'in_tissue', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
        """
        super().__init__(adata, img, pixel_size, meta, cellvit_seg)
        
class XeniumHESTData(HESTData):
    def __init__(self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None,
        xenium_seg: Dict=None
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
                and the following collomns in adata.obs: ['array_col', 'array_row', 'in_tissue', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
            xenium_seg (Dict): path to a xenium nuclei contour file (nucleus_boundaries.csv.gz)
        """
        super().__init__(adata, img, pixel_size, meta, cellvit_seg)
        
        self.xenium_seg = xenium_seg


def read_HESTData(adata_path: str, img: Union[np.ndarray, str], metrics_path: str) -> HESTData:
    adata = sc.read_h5ad(adata_path)
    with open(metrics_path) as metrics_f:     
        metrics = json.load(metrics_f)
    return HESTData(adata, img, metrics['pixel_size_um_estimated'], metrics)
        

def mask_and_patchify_bench(meta_df: pd.DataFrame, save_dir: str, use_mask=True, keep_largest=None):
    i = 0
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        id = row['id']
        img_path = f'/mnt/sdb1/paul/images/pyramidal/{id}.tif'
        adata_path = f'/mnt/sdb1/paul/images/adata/{id}.h5ad'
        metrics_path = os.path.join(get_path_from_meta_row(row), 'processed', 'metrics.json')
        
        hest_obj = read_HESTData(adata_path, img_path, metrics_path)


        keep_largest_args = keep_largest[i] if keep_largest is not None else False

        hest_obj.dump_patches(save_dir,
                           id,
                           verbose=1,
                           use_mask=use_mask,
                           keep_largest=keep_largest_args)
        i += 1
        

def create_benchmark_data(meta_df, save_dir:str, K, adata_folder, use_mask, keep_largest=None):
    os.makedirs(save_dir, exist_ok=True)
    if K is not None:
        splits = meta_df.groupby('patient')['id'].agg(list).to_dict()
        create_splits(os.path.join(save_dir, 'splits'), splits, K=K)
    
    os.makedirs(os.path.join(save_dir, 'patches'), exist_ok=True)
    mask_and_patchify_bench(meta_df, os.path.join(save_dir, 'patches'), use_mask=use_mask, keep_largest=keep_largest)
    
    os.makedirs(os.path.join(save_dir, 'adata'), exist_ok=True)
    for index, row in meta_df.iterrows():
        id = row['id']
        src_adata = os.path.join(adata_folder, id + '.h5ad')
        dst_adata = os.path.join(save_dir, 'adata', id + '.h5ad')
        shutil.copy(src_adata, dst_adata)
        
        
def create_splits(dest_dir, splits, K):
    # [[patient1], [patient2]]...
        

    #meta_df = meta_df[meta_df['id']]
    # [([], []), ] K (nb_split) x 2 x n
    os.makedirs(dest_dir, exist_ok=True)
    
    if K != len(splits):
        print(f'K={K} doesnt match the number of patients, try to distribute the patients instead')
        new_splits = {}
        arr = [value for key, value in splits.items()]
        nb_samples = len([arrss for arrs in arr for arrss in arrs])
        n_per_split = nb_samples // K
        j = 0
        patients = list(splits.keys())
        for i in range(len(patients)):
            new_splits[j] = new_splits.get(j, []) + splits[patients[i]]
            if len(new_splits[j]) >= n_per_split:
                j += 1
        
        splits = new_splits
            
            
    arr = [value for _, value in splits.items()]
    for i in range(len(splits)):
        train_ids = arr.copy()
        del train_ids[i]
        train_ids = [arrss for arrs in train_ids for arrss in arrs]

        test_ids = np.array(arr[i]).flatten()
        print(f'Split {i}/{len(splits)}')
        print('train set is ', train_ids)
        print('')
        print('test set is ', test_ids)
        print('')

        data_train = np.column_stack((train_ids, [os.path.join('patches', id + '.h5') for id in train_ids], [os.path.join('adata', id + '.h5ad') for id in train_ids]))
        train_df = pd.DataFrame(data_train, columns=['sample_id', 'patches_path', 'expr_path'])

        data_test = np.column_stack((test_ids, [os.path.join('patches', id + '.h5') for id in test_ids], [os.path.join('adata', id + '.h5ad') for id in test_ids]))
        test_df = pd.DataFrame(data_test, columns=['sample_id', 'patches_path', 'expr_path'])
        train_df.to_csv(os.path.join(dest_dir, f'train_{i}.csv'), index=False)
        test_df.to_csv(os.path.join(dest_dir, f'test_{i}.csv'), index=False)
        
