import json
import os
import pickle
import shutil
import tempfile
import warnings
from functools import partial
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
try:
    from cucim import CuImage
except ImportError:
    CuImage = None
    print("CuImage is not available. Ensure you have a GPU and cucim installed to use GPU acceleration.")

from hest.segmentation.TissueMask import TissueMask, load_tissue_mask
from hest.wsi import WSI

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

from .segmentation.SegDataset import SegDataset
from .segmentation.segmentation import (apply_otsu_thresholding, keep_largest_area,
                           mask_to_contours, save_pkl, scale_contour_dim,
                           segment_tissue_deep, visualize_tissue_seg)
from .utils import (ALIGNED_HE_FILENAME, check_arg, get_path_from_meta_row,
                    get_path_relative, load_image, plot_verify_pixel_size,
                    tiff_save)
from .vst_save_utils import initsave_hdf5

        
class HESTData:
    """
    Object representing a Spatial Transcriptomics sample along with a full resolution H&E image and metadatas
    """
    
    tissue_mask: np.ndarray = None
    """tissue mask for that sample, will be None until compute_mask() is called"""
    
    contours_tissue: list = None
    """tissue contours for that sample, will be None until compute_mask() is called"""
    
    cellvit_seg = None
    """dictionary of cells in the CellViT .geojson format"""
    
    
    def _verify_format(self, adata):
        assert 'spatial' in adata.obsm
        try:
            adata.uns['spatial']['ST']['images']['downscaled_fullres']
        except KeyError:
            raise ValueError('Downscaled image missing in adata.obs')
        
        features = adata.obs.columns
        required_features = []
        missing = []
        for req in required_features:
            if not req in features:
                missing.append(req)
        if len(missing) > 0:
            raise ValueError(f'The following columns are missing in adata.obs: {missing}')
        
    
    def __init__(
        self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, openslide.OpenSlide, 'CuImage'],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None,
        tissue_seg: TissueMask=None
    ):
        """
        class representing a single ST profile + its associated WSI image
        
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            img (Union[np.ndarray, openslide.OpenSlide, CuImage]): Full resolution image corresponding to the ST data, Openslide/CuImage are lazily loaded, use CuImage for GPU accelerated computation
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
            tissue_seg (TissueMask): tissue mask for that sample
        """
        self.adata = adata
        
        self.wsi = WSI(img)
            
        self.meta = meta
        self._verify_format(adata)
        self.pixel_size = pixel_size
        self.cellvit_seg = cellvit_seg
        self.tissue_mask = None
        self.contours_holes = None
        self.contours_tissue = None
        if tissue_seg is not None:
            self.tissue_mask = tissue_seg.tissue_mask
            self.contours_holes = tissue_seg.contours_holes
            self.contours_tissue = tissue_seg.contours_tissue
        
        if 'total_counts' not in self.adata.var_names:
            sc.pp.calculate_qc_metrics(self.adata, inplace=True)
        
        
    def __repr__(self):
        sup_rep = super().__repr__()

        #img_str = 'WSI in memory'if self.is_image_in_mem() else "WSI not in memory"

        width, height = self.wsi.get_dimensions()
        dim_str = f'WSI has dim height={height}, width={width}'
    
        rep = f"""{sup_rep}
        'pixel_size' is {self.pixel_size}
        {dim_str}
        """
        return rep
        
    
    def save_spatial_plot(self, save_path: str, name: str='', key='total_counts', pl_kwargs={}):
        """Save the spatial plot from that STObject

        Args:
            save_path (str): path to a directory where the spatial plot will be saved
            name (str): save plot as {name}spatial_plots.png
            key (str): feature to plot. Default: 'total_counts'
            pl_kwargs(Dict): arguments for sc.pl.spatial
        """
        print("Plotting spatial plots...")
             
        sc.pl.spatial(self.adata, show=None, img_key="downscaled_fullres", color=[key], title=f"in_tissue spots", **pl_kwargs)
        
        filename = f"{name}spatial_plots.png"
        
        # Save the figure
        plt.savefig(os.path.join(save_path, filename))
        plt.close()  # Close the plot to free memory
        print(f"H&E overlay spatial plots saved in {save_path}")
    
    
    def load_wsi(self) -> None:
        """Load the full WSI in memory"""
        width, height = self.wsi.get_dimensions()
        self.wsi = WSI(self.wsi.get_thumbnail(width, height))
    
        
    def save(self, path: str, save_img=True, pyramidal=True, bigtiff=False, plot_pxl_size=False):
        """Save a HESTData object to `path` as follows:
            - aligned_adata.h5ad (contains expressions for each spots + their location on the fullres image + a downscaled version of the fullres image)
            - metrics.json (contains useful metrics)
            - downscaled_fullres.jpeg (a downscaled version of the fullres image)
            - aligned_fullres_HE.tif (the full resolution image)
            - cells.geojson (cell segmentation if it exists)
            - Optional: tissue_mask.jpg (grayscale image of the tissue segmentation if it exists)
            - Optional: tissue_mask.pkl (main contours and contours of holes of the tissue segmentation if it exists)
            - Optional: tissue_seg_vis.jpg (visualization of tissue contour and holes on downscaled H&E if it exists)

        Args:
            path (str): save location
            save_img (bool): whenever to save the image at all (can save a lot of time if set to False). Defaults to True
            pyramidal (bool, optional): whenever to save the full resolution image as pyramidal (can be slow to save, however it's sometimes necessary for loading large images in QuPath). Defaults to True.
            bigtiff (bool, optional): whenever the bigtiff image is more than 4.1GB. Defaults to False.
        """
        try:
            self.adata.write(os.path.join(path, 'aligned_adata.h5ad'))
        except:
            # workaround from https://github.com/theislab/scvelo/issues/255
            self.adata.__dict__['_raw'].__dict__['_var'] = self.adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
            self.adata.write(os.path.join(path, 'aligned_adata.h5ad'))
        
        if save_img:
            img = self.wsi.numpy()
        self.meta['adata_nb_col'] = len(self.adata.var_names)
        
        width, height = self.wsi.get_dimensions()
        
        self.meta['fullres_px_width'] = width
        self.meta['fullres_px_height'] = height
        with open(os.path.join(path, 'metrics.json'), 'w') as json_file:
            json.dump(self.meta, json_file) 
            
        if self.cellvit_seg is not None:
            with open(os.path.join(path, 'cells.geojson'), 'w') as json_file:
                json.dump(self.cellvit_seg, json_file)
        
        downscaled_img = self.adata.uns['spatial']['ST']['images']['downscaled_fullres']
        down_fact = self.adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
        down_img = Image.fromarray(downscaled_img)
        down_img.save(os.path.join(path, 'downscaled_fullres.jpeg'))
        
        
        if plot_pxl_size:
            pixel_size_embedded = self.meta['pixel_size_um_embedded']
            pixel_size_estimated = self.meta['pixel_size_um_estimated']
            
            
            plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, os.path.join(path, 'pixel_size_vis.png'))


        if self.tissue_mask is not None:
            Image.fromarray(self.tissue_mask).save(os.path.join(path, 'tissue_mask.jpg'))
            self.save_tissue_seg_pkl(path, 'tissue')
            
            vis = visualize_tissue_seg(
                    self.wsi.img,
                    self.tissue_mask,
                    self.contours_tissue,
                    self.contours_holes,
                    line_color=(0, 255, 0),
                    hole_color=(0, 0, 255),
                    line_thickness=5,
                    target_width=1000,
                    seg_display=True,
            )
            
            vis.save(os.path.join(path, 'tissue_seg_vis.jpg'))

        
        if save_img:
            tiff_save(img, os.path.join(path, ALIGNED_HE_FILENAME), self.pixel_size, pyramidal=pyramidal, bigtiff=bigtiff)


    def compute_mask(
        self, 
        keep_largest=False, 
        thumbnail_width=2000, 
        method: str='deep', 
        batch_size=8, 
        model_name='deeplabv3_seg_v4.ckpt'
    ) -> None:
        """ Compute tissue mask and stores it in the current HESTData object

        Args:
            method (str, optional): perform deep learning based segmentation ('deep') or otsu based ('otsu').
                Deep-learning based segmentation will be more accurate but a GPU is recommended, 'otsu' is faster but less accurate. Defaults to 'deep'.
            batch_size (int, optional): inference batch_size if method=`deep` is selected. Defaults to 8.
            model_name (str, optional): name of model weights in `models` used if method=`deep` is selected. Defaults to 'deeplabv3_seg_v4.ckpt'.
        """
        
        check_arg(method, 'method', ['deep', 'otsu'])
        
        if method == 'deep':
            self.tissue_mask, self.contours_tissue, self.contours_holes = segment_tissue_deep(self.wsi, self.pixel_size, target_pxl_size=1, patch_size=512, batch_size=batch_size, model_name=model_name)
        elif method == 'otsu':
        
            width, height = self.wsi.get_dimensions()
            scale = thumbnail_width / width
            thumbnail = self.wsi.get_thumbnail(round(width * scale), round(height * scale))
            mask = apply_otsu_thresholding(thumbnail).astype(np.uint8)
            mask = 1 - mask
            if keep_largest:
                mask = keep_largest_area(mask)
            self.tissue_mask = np.round(cv2.resize(mask, (width, height))).astype(np.uint8)
            self.contours_tissue, self.contours_holes = mask_to_contours(self.tissue_mask, pixel_size=self.pixel_size)


    def get_tissue_mask(self, keep_largest=False, method: str='deep') -> np.ndarray:
        """ Return existing tissue segmentation mask if it exists, implicitly compute it beforehand if it doesn't exists

        Args:
            method (str, optional): perform deep learning based segmentation ('deep') or otsu based ('otsu').
            Deep-learning based segmentation will be more accurate but a GPU is recommended, 'otsu' is faster but less accurate. Defaults to 'deep'.

        Returns:
            np.ndarray: an array with the same resolution as the WSI image, where 1 means tissue and 0 means background
        """
        
        if self.tissue_mask is None:
            self.compute_mask(keep_largest=keep_largest, method=method)
        return self.tissue_mask
    

    def dump_patches(
        self,
        patch_save_dir: str,
        name: str = 'patches',
        target_patch_size: int=224,
        target_pixel_size: float=0.5,
        verbose=0,
        dump_visualization=True,
        use_mask=True,
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
        img_width, img_height = self.wsi.get_dimensions()
        patch_rectangles = [] # lower corner (x, y) + (widht, height)
        downscale_vis = TARGET_VIS_SIZE / img_width

        if use_mask:
            tissue_mask = self.get_tissue_mask(keep_largest)
        else:
            tissue_mask = np.ones((img_height, img_width)).astype(np.uint8)
            
        mask_plot = visualize_tissue_seg(
                        self.wsi.img,
                        self.tissue_mask,
                        self.contours_tissue,
                        self.contours_holes
                )

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
            
            image_patch = self.wsi.read_region((xImage - patch_size_pxl // 2, yImage - patch_size_pxl // 2), (patch_size_pxl, patch_size_pxl))
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
            assert image_patch.shape == (target_patch_size, target_patch_size, 3)
            asset_dict = { 'img': np.expand_dims(image_patch, axis=0),  # (1 x w x h x 3)
                            'coords': np.expand_dims([yImage, xImage], axis=0),   # (1 x 2)
                            'barcode': np.expand_dims([barcode_spot], axis=0)
                            }

            attr_dict = {}
            attr_dict['img'] = {'patch_size': patch_size_pxl,
                                'factor': scale_factor}

            initsave_hdf5(output_datafile, asset_dict, attr_dict, mode=mode_HE)
            mode_HE = 'a'

        
        if dump_visualization:
            ax.add_collection(PatchCollection(patch_rectangles, facecolor='none', edgecolor='black', linewidth=0.3))
            ax.set_axis_off()
            os.makedirs(os.path.join(patch_save_dir, 'vis'), exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(patch_save_dir, 'vis', name + '.png'), dpi=400, bbox_inches = 'tight')
            
        if verbose:
            print(f'found {patch_count} valid patches')
            
    
    def __verify_mask(self):
        if self.tissue_mask is None:
            raise Exception("compute the tissue mask for this HESTData object with self.compute()")        
    

    def get_tissue_contours(self) -> Dict[str, list]:
        """Get the tissue contours and holes

        Returns:
            Dict[str, list]: dictionnary of contours and holes in the tissue
        """
        
        self.__verify_mask()
        
        asset_dict = {'holes': self.contours_holes, 
                      'tissue': self.contours_tissue, 
                      'groups': None}
        return asset_dict
    

    def save_tissue_seg_jpg(self, save_dir: str, name: str = 'hest') -> None:
        """Save tissue segmentation as a greyscale .jpg file, downscale the tissue mask such that the width 
        and the height are less than 40k pixels

        Args:
            save_dir (str): path to save directory
            name (str): .jpg file is saved as {name}_mask.jpg
        """
        
        self.__verify_mask()
        
        MAX_EDGE = 40000
        
        longuest_edge = max(self.tissue_mask.shape[0], self.tissue_mask.shape[1])
        img = self.tissue_mask
        if longuest_edge > MAX_EDGE:
            downscaled = MAX_EDGE / longuest_edge
            width, height = self.tissue_mask.shape[1], self.tissue_mask.shape[0]
            img = cv2.resize(img, (round(downscaled * width), round(downscaled * height)))
        
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f'{name}_mask.jpg'))
        
            
    def save_tissue_seg_pkl(self, save_dir: str, name: str) -> None:
        """Save tissue segmentation contour as a .pkl file

        Args:
            save_dir (str): path to pkl file
            name (str): .pkl file is saved as {name}_mask.pkl
        """
        
        self.__verify_mask()

        asset_dict = self.get_tissue_contours()
        save_pkl(os.path.join(save_dir, f'{name}_mask.pkl'), asset_dict)
        
    
    def save_vis(self, save_dir, name) -> None:
        
        vis = visualize_tissue_seg(
            self.wsi.img,
            self.tissue_mask,
            self.contours_tissue,
            self.contours_holes,
            line_color=(0, 255, 0),
            hole_color=(0, 0, 255),
            line_thickness=5,
            target_width=1000,
            seg_display=True,
        )
        vis.save(os.path.join(save_dir, f'{name}_vis.jpg'))


    def to_spatial_data(self, lazy_img=True) -> SpatialData:
        """Convert a HESTData sample to a scverse SpatialData object
        
        Args:
            lazy_img (bool, optional): whenever to lazily load the image if not already loaded (e.g. self.wsi is of type OpenSlide or CuImage). Defaults to True.

        Returns:
            SpatialData: scverse SpatialData object
        """
        
        def read_hest_wsi(path):
            return pyvips.Image.tiffload(path).numpy().transpose((2, 0, 1))
    
        if lazy_img and not (isinstance(self.wsi, np.ndarray)):
            
            with tifffile.TiffFile(self.wsi) as tif:
                page = tif.pages[0]
                width = page.imagewidth
                height = page.imagelength
                
            
            img = from_delayed(delayed(read_hest_wsi)(self.wsi), shape=(height, width, 3), dtype=np.int8)
        else:
            img = self.wsi.numpy()
            arr = da.from_array(img)
            sp_img = SpatialImage(arr, dims=['c', 'y', 'x'], attrs={'transform': None})
        
        st = SpatialData({'fullres': sp_img}, table=self.adata)
        
        #TODO add CellViT
        #TODO add tissue segmentation
        
        return st
    

class VisiumHESTData(HESTData): 
    def __init__(self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None,
        tissue_seg: TissueMask=None
    ):
        super().__init__(adata, img, pixel_size, meta, cellvit_seg, tissue_seg)

class VisiumHDHESTData(HESTData): 
    def __init__(self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None,
        tissue_seg: TissueMask=None
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
            tissue_seg (TissueMask): tissue mask for that sample
        """
        super().__init__(adata, img, pixel_size, meta, cellvit_seg, tissue_seg)        
        
class STHESTData(HESTData):
    def __init__(self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None,
        tissue_seg: TissueMask=None
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
            tissue_seg (TissueMask): tissue mask for that sample
        """
        super().__init__(adata, img, pixel_size, meta, cellvit_seg, tissue_seg)
        
class XeniumHESTData(HESTData):

    def __init__(
        self, 
        adata: sc.AnnData,
        img: Union[np.ndarray, openslide.OpenSlide, 'CuImage'],
        pixel_size: float,
        meta: Dict = {},
        cellvit_seg: Dict=None,
        tissue_seg: TissueMask=None,
        xenium_nuc_seg: pd.DataFrame=None,
        xenium_cell_seg: pd.DataFrame=None,
        cell_adata: sc.AnnData=None,
        transcript_df: pd.DataFrame=None
    ):
        """
        class representing a single ST profile + its associated WSI image
        
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object (pooled by patch for Xenium)
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            img (Union[np.ndarray, openslide.OpenSlide, CuImage]): Full resolution image corresponding to the ST data, Openslide/CuImage are lazily loaded, use CuImage for GPU accelerated computation
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            cellvit_seg (Dict): dictionary of cells in the CellViT .geojson format. Default: None
            tissue_seg (TissueMask): tissue mask for that sample
            xenium_nuc_seg (pd.DataFrame): content of a xenium nuclei contour file as a dataframe (nucleus_boundaries.parquet)
            xenium_cell_seg (pd.DataFrame): content of a xenium cell contour file as a dataframe (cell_boundaries.parquet)
            cell_adata (sc.AnnData): ST cell data, each row in adata.obs is a cell, each row in obsm is the cell location on the H&E image in pixels
            transcript_df (pd.DataFrame): dataframe of transcripts, each row is a transcript, he_x and he_y is the transcript location on the H&E image in pixels
        """
        super().__init__(adata=adata, img=img, pixel_size=pixel_size, meta=meta, cellvit_seg=cellvit_seg, tissue_seg=tissue_seg)
        
        self.xenium_nuc_seg = xenium_nuc_seg
        self.xenium_cell_seg = xenium_cell_seg
        self.cell_adata = cell_adata
        self.transcript_df = transcript_df
        
        
    def save(self, path: str, save_img=True, pyramidal=True, bigtiff=False, plot_pxl_size=False):
        """Save a HESTData object to `path` as follows:
            - aligned_adata.h5ad (contains expressions for each spots + their location on the fullres image + a downscaled version of the fullres image)
            - metrics.json (contains useful metrics)
            - downscaled_fullres.jpeg (a downscaled version of the fullres image)
            - aligned_fullres_HE.tif (the full resolution image)
            - cells.geojson (cell segmentation if it exists)
            - Optional: cells_xenium.geojson (if xenium cell segmentation is attached to this object)
            - Optional: nuclei_xenium.geojson (if xenium cell segmentation is attached to this object)
            - Optional: tissue_mask.jpg (grayscale image of the tissue segmentation if it exists)
            - Optional: tissue_mask.pkl (main contours and contours of holes of the tissue segmentation if it exists)

        Args:
            path (str): save location
            save_img (bool): whenever to save the image at all (can save a lot of time if set to False)
            pyramidal (bool, optional): whenever to save the full resolution image as pyramidal (can be slow to save, however it's sometimes necessary for loading large images in QuPath). Defaults to True.
            bigtiff (bool, optional): whenever the bigtiff image is more than 4.1GB. Defaults to False.
        """
        super().save(path, save_img, pyramidal, bigtiff, plot_pxl_size)
        if self.cell_adata is not None:
            self.cell_adata.write_h5ad(os.path.join(path, 'aligned_cells.h5ad'))
        
        if self.transcript_df is not None:
            self.transcript_df.to_parquet(os.path.join(path, 'aligned_transcripts.parquet'))
            
        if self.xenium_nuc_seg is not None:
            print('Saving Xenium nucleus boundaries... (can be slow)')
            with open(os.path.join(path, 'nuclei_xenium.geojson'), 'w') as f:
                json.dump(self.xenium_nuc_seg, f, indent=4)
                
        if self.xenium_cell_seg is not None:
            print('Saving Xenium cells boundaries... (can be slow)')
            with open(os.path.join(path, 'cells_xenium.geojson'), 'w') as f:
                json.dump(self.xenium_cell_seg, f, indent=4)
            
            
        # TODO save segmentation
        


def read_HESTData(
    adata_path: str, 
    img: Union[str, np.ndarray, openslide.OpenSlide, 'CuImage'], 
    metrics_path: str,
    mask_path_pkl: str = None,
    mask_path_jpg: str = None
) -> HESTData:
    """ Read a HEST sample from disk

    Args:
        adata_path (str): path to .h5ad adata file containing ST data the 
            adata object must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
        img (Union[str, np.ndarray, openslide.OpenSlide, CuImage]): path to a full resolution image (if passed as str) or full resolution image corresponding to the ST data, Openslide/CuImage are lazily loaded, use CuImage for GPU accelerated computation
        pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
        metrics_path (str): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
        mask_path_pkl (str): path to a .pkl file containing the tissue segmentation contours. Defaults to None.
        mask_path_jpg (str): path to a .jog file containing the greyscale tissue segmentation mask. Defaults to None.

    Returns:
        HESTData: HESTData object
    """
    
    if isinstance(img, str):
        if CuImage is not None:
            img = CuImage(img)
            width, height = img.resolutions['level_dimensions'][0]
        else:
            img = openslide.OpenSlide(img)
            width, height = img.dimensions
            
            
    if mask_path_pkl is not None and mask_path_jpg is not None:
        tissue_seg = load_tissue_mask(mask_path_pkl, mask_path_jpg, width, height)
    
    
    adata = sc.read_h5ad(adata_path)
    with open(metrics_path) as metrics_f:     
        metrics = json.load(metrics_f)
    return HESTData(adata, img, metrics['pixel_size_um_estimated'], metrics, tissue_seg=tissue_seg)
        

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
        

def load_hest(hest_dir: str) -> List[HESTData]:
    """Read HESTData objects from a local directory

    Args:
        hest_dir (str): hest directory containing folders: st, wsis, metadata, tissue_seg (optional)

    Returns:
        List[HESTData]: list of HESTData objects
    """
    
    ## TODO also read mask

    hestdata_list = []
    for st_filename in os.listdir(os.path.join(hest_dir, 'st')):
        id = st_filename.split('.')[0]
        adata_path = os.path.join(hest_dir, 'st', f'{id}.h5ad')
        img_path = os.path.join(hest_dir, 'wsis', f'{id}.tif')
        meta_path = os.path.join(hest_dir, 'metadata', f'{id}.json')
        masks_path_pkl = None
        masks_path_jpg = None
        if os.path.exists(os.path.join(hest_dir, 'tissue_seg')):
            masks_path_pkl = os.path.join(hest_dir, 'tissue_seg', f'{id}_mask.pkl')
            masks_path_jpg = os.path.join(hest_dir, 'tissue_seg', f'{id}_mask.jpg')
        st = read_HESTData(adata_path, img_path, meta_path, masks_path_pkl, masks_path_jpg)
        hestdata_list.append(st)
    return hestdata_list