from __future__ import annotations

import json
import os
import shutil
import warnings
from typing import Dict, List, Union

import cv2
import geopandas as gpd
import numpy as np
from loguru import logger
from hestcore.wsi import (WSI, CucimWarningSingleton, NumpyWSI,
                          contours_to_img, wsi_factory)
from loguru import logger

from hest.io.seg_readers import TissueContourReader, write_geojson
from hest.LazyShapes import LazyShapes, convert_old_to_gpd, old_geojson_to_new
from hest.segmentation.TissueMask import TissueMask, load_tissue_mask

try:
    import openslide
except Exception:
    print("Couldn't import openslide, verify that openslide is installed on your system, https://openslide.org/download/")
import pandas as pd
from hestcore.segmentation import (apply_otsu_thresholding, mask_to_gdf,
                                   save_pkl, segment_tissue_deep, get_path_relative)
from PIL import Image
from shapely import Point
from tqdm import tqdm

from .utils import (ALIGNED_HE_FILENAME, check_arg, deprecated,
                    find_first_file_endswith, get_k_genes_from_df, get_path_from_meta_row,
                    plot_verify_pixel_size, tiff_save, verify_paths)


class HESTData:
    """
    Object representing a Spatial Transcriptomics sample along with a full resolution H&E image and metadatas
    """
    
    shapes: List[LazyShapes] = []
    """ List of `LazyShapes`, i.e. cells, nuclei """

    
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
        adata: sc.AnnData, # type: ignore
        img: Union[np.ndarray, openslide.OpenSlide, CuImage, str], # type: ignore
        pixel_size: float,
        meta: Dict = {},
        tissue_seg: TissueMask=None,
        tissue_contours: gpd.GeoDataFrame=None,
        shapes: List[LazyShapes]=[]
    ):
        """
        class representing a single ST profile + its associated WSI image
        
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            img (Union[np.ndarray, openslide.OpenSlide, CuImage, str]): Full resolution image corresponding to the ST data, Openslide/CuImage are lazily loaded, use CuImage for GPU accelerated computation. 
                If a str is passed, the image is opened with cucim if available and OpenSlide otherwise
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            shapes (List[LazyShapes]): dictionary of shapes, note that these shapes will be lazily loaded. Default: []
            tissue_seg (TissueMask): *Deprecated* tissue mask for that sample
        """
        import scanpy as sc
        
        self.adata = adata
        
        self.wsi = wsi_factory(img)
            
        self.meta = meta
        self._verify_format(adata)
        self.pixel_size = pixel_size
        self.shapes = shapes
        if tissue_seg is not None:
            warnings.warn('tissue_seg is deprecated, please use tissue_contours instead, you might have to delete and redownload the `tissue_seg` data directory from huggingface')
            self._tissue_contours = convert_old_to_gpd(tissue_seg.contours_holes, tissue_seg.contours_tissue)
        else:
            self._tissue_contours = tissue_contours
        
        if 'total_counts' not in self.adata.var_names and len(self.adata) > 0:
            sc.pp.calculate_qc_metrics(self.adata, inplace=True)
        
        
    def __repr__(self):
        sup_rep = super().__repr__()
    
        rep =  f"""{sup_rep}
        'pixel_size' is {self.pixel_size}
        'wsi' is {self.wsi}
        'shapes': {self.shapes}"""
        
        return rep
        
    
    def save_spatial_plot(self, save_path: str, name: str='', key='total_counts', pl_kwargs={}):
        """Save the spatial plot from that STObject

        Args:
            save_path (str): path to a directory where the spatial plot will be saved
            name (str): save plot as {name}spatial_plots.png
            key (str): feature to plot. Default: 'total_counts'
            pl_kwargs(Dict): arguments for sc.pl.spatial
        """
        
        save_spatial_plot(self.adata, save_path, name, key, pl_kwargs)
    
    
    def load_wsi(self) -> None:
        """Load the full WSI in memory"""
        self.wsi = NumpyWSI(self.wsi.numpy())
    
        
    def save(self, path: str, save_img=True, pyramidal=True, bigtiff=False, plot_pxl_size=False, **kwargs):
        """Save a HESTData object to `path` as follows:
            - aligned_adata.h5ad (contains expressions for each spots + their location on the fullres image + a downscaled version of the fullres image)
            - metrics.json (contains useful metrics)
            - downscaled_fullres.jpeg (a downscaled version of the fullres image)
            - aligned_fullres_HE.tif (the full resolution image)
            - cells.geojson (cell segmentation if it exists)
            - Optional: tissue_contours.geojson (contours of the tissue segmentation if it exists)
            - Optional: tissue_seg_vis.jpg (visualization of tissue contour and holes on downscaled H&E if it exists)

        Args:
            path (str): save location
            save_img (bool): whenever to save the image at all (can save a lot of time if set to False). Defaults to True
            pyramidal (bool, optional): whenever to save the full resolution image as pyramidal (can be slow to save, however it's sometimes necessary for loading large images in QuPath). Defaults to True.
            bigtiff (bool, optional): whenever the bigtiff image is more than 4.1GB. Defaults to False.
        """
        os.makedirs(path, exist_ok=True)
        
        try:
            self.adata.write(os.path.join(path, 'aligned_adata.h5ad'))
        except:
            # workaround from https://github.com/theislab/scvelo/issues/255
            import traceback
            traceback.print_exc()
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
        
        downscaled_img = self.adata.uns['spatial']['ST']['images']['downscaled_fullres']
        down_fact = self.adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
        down_img = Image.fromarray(downscaled_img)
        if len(downscaled_img) > 0:
            down_img.save(os.path.join(path, 'downscaled_fullres.jpeg'))
        
        
        if plot_pxl_size:
            pixel_size_embedded = self.meta['pixel_size_um_embedded']
            pixel_size_estimated = self.meta['pixel_size_um_estimated']
            
            
            plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, os.path.join(path, 'pixel_size_vis.png'))


        if self._tissue_contours is not None:
            self.save_tissue_contours(path, 'tissue')         
            self.save_tissue_vis(path, 'tissue_seg')

        
        if save_img:
            tiff_save(img, os.path.join(path, ALIGNED_HE_FILENAME), self.pixel_size, pyramidal=pyramidal, bigtiff=bigtiff)


    def segment_tissue(
        self,
        fast_mode=False,
        target_pxl_size=1,
        patch_size_um=512,
        model_name='deeplabv3_seg_v4.ckpt',
        batch_size=8,
        auto_download=True,
        num_workers=8,
        thumbnail_width=2000, 
        method: str='deep',
        weights_dir = None
    ) -> Union[None, np.ndarray]:
        """ Compute tissue mask and stores it in the current HESTData object

        Args:
            fast_mode (bool, optional): in fast mode the inference is done at 2 um/px instead of 1 um/px, 
                note that the inference pixel size is overwritten by the `target_pxl_size` argument if != 1. Defaults to False.
            target_pxl_size (int, optional): patches are scaled to this pixel size in um/px for inference. Defaults to 1.
            patch_size_um (int, optional): patch size in um. Defaults to 512.
            model_name (str, optional): model name in `HEST/models` dir. Defaults to 'deeplabv3_seg_v4.ckpt'.
            batch_size (int, optional): batch size for inference. Defaults to 8.
            auto_download (bool, optional): whenever to download the model weights automatically if not found. Defaults to True.
            num_workers (int, optional): number of workers for the dataloader during inference. Defaults to 8.
            thumbnail_width (int, optional): size at which otsu segmentation is performed, ignored if method is 'deep'
            method (str, optional): perform deep learning based segmentation ('deep') or otsu based ('otsu').
                Deep-learning based segmentation will be more accurate but a GPU is recommended, 'otsu' is faster but less accurate. Defaults to 'deep'.
            weights_dir (str, optional): directory containing the models, if None will be ../models relative to the src package of hestcore. None
                
        Returns:
            gpd.GeoDataFrame: a geodataframe of the tissue contours, contains a column `tissue_id` indicating to which tissue the contour belongs to.
        """
        
        check_arg(method, 'method', ['deep', 'otsu'])
        
        if method == 'deep':
            self._tissue_contours = segment_tissue_deep(
                self.wsi,
                self.pixel_size,
                fast_mode,
                target_pxl_size,
                patch_size_um,
                model_name,
                batch_size,
                auto_download,
                num_workers,
                weights_dir
            )
        elif method == 'otsu':
        
            width, height = self.wsi.get_dimensions()
            scale = thumbnail_width / width
            thumbnail = self.wsi.get_thumbnail(round(width * scale), round(height * scale))
            mask = apply_otsu_thresholding(thumbnail).astype(np.uint8)
            mask = 1 - mask
            tissue_mask = np.round(cv2.resize(mask, (width, height))).astype(np.uint8)
            
            #TODO directly convert to gpd
            gdf_contours = mask_to_gdf(tissue_mask, pixel_size=self.pixel_size)
            self._tissue_contours = gdf_contours
            
        return self.tissue_contours
    
    
    def save_tissue_contours(self, save_dir: str, name: str) -> None:
        self.tissue_contours.to_file(os.path.join(save_dir, name + '_contours.geojson'), driver="GeoJSON")   

    @deprecated
    def get_tissue_mask(self) -> np.ndarray:
        """ Deprecated. Return existing tissue segmentation mask if it exists, raise an error if it doesn't exist

        Returns:
            np.ndarray: an array with the same resolution as the WSI image, where 1 means tissue and 0 means background
        """
        
        self.__verify_mask()
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
        threshold=0.15
    ):
        """ Dump H&E patches centered around ST spots to a .h5 file. 
        
            Patches are computed such that:
             - each patch is rescaled to `target_pixel_size` um/px
             - a crop of `target_patch_size`x`target_patch_size` pixels around each ST (pseudo) spot is derived (which coordinates are derived from adata.obsm['spatial'])

        Args:
            patch_save_dir (str): directory where the .h5 patch file will be saved
            name (str, optional): file will be saved as {name}.h5. Defaults to 'patches'.
            target_patch_size (int, optional): target patch size in pixels (after scaling to match `target_pixel_size`). Defaults to 224.
            target_pixel_size (float, optional): target patch pixel size in um/px. Defaults to 0.5.
            verbose (int, optional): verbose. Defaults to 0.
            dump_visualization (bool, optional): whenever to dump a visualization of the patches on top of the downscaled WSI. Defaults to True.
            use_mask (bool, optional): whenever to take into account the tissue mask. Defaults to True.
            threshold (float, optional): Tissue intersection threshold for a patch to be kept. Defaults to 0.15
        """
        
        os.makedirs(patch_save_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        dst_pixel_size = target_pixel_size
        
        adata = self.adata.copy()
        
        for index in adata.obs.index:
            if len(index) != len(adata.obs.index[0]):
                warnings.warn("indices of adata.obs should all have the same length to avoid problems when saving to h5", UserWarning)
                
        
        src_pixel_size = self.pixel_size
        
        patch_count = 0
        h5_path = os.path.join(patch_save_dir, name + '.h5')

        assert len(adata.obs) == len(adata.obsm['spatial'])
        
        patch_size_src = target_patch_size * (dst_pixel_size / src_pixel_size)
        coords_center = adata.obsm['spatial']
        coords_topleft = coords_center - patch_size_src // 2
        len_tmp = len(coords_topleft)
        in_slide_mask = (0 <= coords_topleft[:, 0] + patch_size_src) & (coords_topleft[:, 0] < self.wsi.width) & (0 <= coords_topleft[:, 1] + patch_size_src) & (coords_topleft[:, 1] < self.wsi.height)
        coords_topleft = coords_topleft[in_slide_mask]
        if len(coords_topleft) < len_tmp:
            warnings.warn(f"Filtered {len_tmp - len(coords_topleft)} spots outside the WSI")
        
        barcodes = np.array(adata.obs.index)
        barcodes = barcodes[in_slide_mask]
        mask = self.tissue_contours if use_mask else None
        coords_topleft = np.array(coords_topleft).astype(int)
        patcher = self.wsi.create_patcher(target_patch_size, src_pixel_size, dst_pixel_size, mask=mask, custom_coords=coords_topleft, threshold=threshold)

        if mask is not None:
            valid_barcodes = barcodes[patcher.valid_mask]

        patcher.to_h5(h5_path, extra_assets={'barcode': valid_barcodes})

        if dump_visualization:
            patcher.save_visualization(os.path.join(patch_save_dir, name + '_patch_vis.png'), dpi=400)
            
        if verbose:
            print(f'found {patch_count} valid patches')
            
    
    def __verify_mask(self):
        if self.tissue_contours is None:
            raise Exception("No existing tissue mask for that sample, compute the tissue mask with self.segment_tissue()")        
    
    
    def get_shapes(self, name, coordinate_system):
        for shape in self.shapes:
            if shape.name == name and shape.coordinate_system == coordinate_system:
                return shape
        return None
    

    @deprecated
    def get_tissue_contours(self) -> Dict[str, list]:
        """*Deprecated* use `self.tissue_contours` instead. 
        
        Get the tissue contours and holes

        Returns:
            Dict[str, list]: dictionnary of contours and holes in the tissue
        """
        
        self.__verify_mask()
        

        contours_tissue = self.tissue_contours.geometry.values
        contours_tissue = [list(c.exterior.coords) for c in contours_tissue]
        contours_holes = [[] for _ in range(len(contours_tissue))]
        
        
        asset_dict = {'holes': contours_holes, 
                      'tissue': contours_tissue, 
                      'groups': None}
        return asset_dict
    
    
    @property
    def tissue_contours(self) -> gpd.GeoDataFrame:
        """ Geodataframe of tissue contours polygons also contains a tissue_id column """
        if self._tissue_contours is None:
            raise Exception("No tissue segmentation attached to this sample, segment tissue first by calling `segment_tissue()` for this object")
        return self._tissue_contours

    @deprecated
    def save_tissue_seg_jpg(self, save_dir: str, name: str = 'hest') -> None:
        """*Deprecated* Save tissue segmentation as a greyscale .jpg file, downscale the tissue mask such that the width 
        and the height are less than 40k pixels

        Args:
            save_dir (str): path to save directory
            name (str): .jpg file is saved as {name}_mask.jpg
        """
        
        self.__verify_mask()
        
        img_width, img_height = self.wsi.get_dimensions()
        tissue_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        tissue_mask = contours_to_img(
                self.tissue_contours, 
                tissue_mask, 
                fill_color=(1, 1, 1)
        )[:, :, 0]
        
        MAX_EDGE = 40000
        
        longuest_edge = max(tissue_mask.shape[0], tissue_mask.shape[1])
        img = tissue_mask
        if longuest_edge > MAX_EDGE:
            downscaled = MAX_EDGE / longuest_edge
            width, height = tissue_mask.shape[1], tissue_mask.shape[0]
            img = cv2.resize(img, (round(downscaled * width), round(downscaled * height)))
        
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f'{name}_mask.jpg'))
        
    
    @deprecated
    def save_tissue_seg_pkl(self, save_dir: str, name: str) -> None:
        """*Deprecated* Save tissue segmentation contour as a .pkl file

        Args:
            save_dir (str): path to pkl file
            name (str): .pkl file is saved as {name}_mask.pkl
        """
        
        self.__verify_mask()

        asset_dict = self.get_tissue_contours()
        save_pkl(os.path.join(save_dir, f'{name}_mask.pkl'), asset_dict)
        
    
    def get_tissue_vis(self):
         return self.wsi.get_tissue_vis(
            self.tissue_contours,
            fill_color=(0, 255, 0),
            target_width=1000,
            seg_display=True,
        )
    
    
    @deprecated
    def save_vis(self, save_dir, name) -> None:
        """ *Deprecated* use save_tissue_vis instead"""
        vis = self.get_tissue_vis()
        vis.save(os.path.join(save_dir, f'{name}_vis.jpg'))
        
    def save_tissue_vis(self, save_dir: str, name: str) -> None:
        """ Save a visualization of the tissue segmentation on top of the downscaled H&E

        Args:
            save_dir (str): directory where the visualization will be saved
            name (str): file is saved as {save_dir}/{name}_vis.jpg
        """
        vis = self.get_tissue_vis()
        vis.save(os.path.join(save_dir, f'{name}_vis.jpg'))

    def to_spatial_data(self, fullres: bool = False) -> SpatialData: 
        """
        Convert a HESTData sample to a scverse SpatialData object. Note that a large part of this function is based on 
        spatialdata-io's [``from_legacy_anndata``](https://spatialdata.scverse.org/projects/io/en/latest/generated/spatialdata_io.experimental.from_legacy_anndata.html) 
        function with some adjustments for ``HESTData``. 

        Args:
            fullres (bool, optional): Includes pyramidal full resolution whole slide image as a ``DataTree`` object for those dimensions compatible with 
                Image2DModel's downsampling. Defaults to False.

        Returns:
            SpatialData: scverse SpatialData oobject containing the ``hires`` and ``lowres`` downsampled versions
                of the image and their respective coordinate systems.  
            
        Example: 
            ```python
            from hest import load_hest
            hest_data = load_hest('../hest_data', id_list=['TENX68'])
            st = hest_data[0]
            st.to_spatial_data(fullres=True)

            >>>
            
            ```
            SpatialData object
            ├── Images
            │     ├── 'ST_downscaled_hires_image': SpatialImage[cyx] (3, 4779, 2586)
            │     ├── 'ST_downscaled_lowres_image': SpatialImage[cyx] (3, 1000, 541)
            │     └── 'ST_fullres_image': DataTree[cyx] (3, 38232, 20690), (3, 19116, 10345)
            ├── Shapes
            │     └── 'locations': GeoDataFrame shape: (1657, 2) (2D shapes)
            └── Tables
                └── 'table': AnnData (1657, 18085)
            with coordinate systems:
                ▸ 'ST_downscaled_hires', with elements:
                    ST_downscaled_hires_image (Images), locations (Shapes)
                ▸ 'ST_downscaled_lowres', with elements:
                    ST_downscaled_lowres_image (Images), locations (Shapes)
                ▸ 'ST_fullres', with elements:
                    ST_fullres_image (Images), locations (Shapes)
            ```
            
        """
        
        # imports specific to spatial data conversion
        from dask import delayed
        from dask.array import from_delayed
        from spatial_image import SpatialImage
        from spatialdata import SpatialData
        from spatialdata.models import Image2DModel, ShapesModel, TableModel
        from spatialdata.transformations import Identity, Scale
        
        # AnnData keys
        SPATIAL = "spatial"
        SCALEFACTORS = "scalefactors"
        TISSUE_HIRES_SCALEF = "tissue_hires_scalef"
        TISSUE_LOWRES_SCALEF = "tissue_downscaled_fullres_scalef"
        SPOT_DIAMETER_FULLRES = "spot_diameter_fullres"

        IMAGES = "images"
        HIRES = "fullres"
        LOWRES = "downscaled_fullres"

        # SpatialData keys
        REGION = "locations"
        REGION_KEY = "region"
        INSTANCE_KEY = "instance_id"
        SPOT_DIAMETER_FULLRES_DEFAULT = 10
        
        images = {}
        shapes = {}
        spot_diameter_fullres_list = []
        shapes_transformations = {}
        if SPATIAL in self.adata.uns:
            dataset_ids = list(self.adata.uns[SPATIAL].keys())
            for dataset_id in dataset_ids:
                # read the image data and the scale factors for the shapes
                keys = set(self.adata.uns[SPATIAL][dataset_id].keys())
                tissue_hires_scalef = None
                tissue_lowres_scalef = None
                hires = None
                lowres = None
                if SCALEFACTORS in keys:
                    scalefactors = self.adata.uns[SPATIAL][dataset_id][SCALEFACTORS]
                    if TISSUE_HIRES_SCALEF in scalefactors:
                        tissue_hires_scalef = scalefactors[TISSUE_HIRES_SCALEF]
                    else: 
                        pixel_size=self.meta['pixel_size_um_estimated']
                        ds_factor = 4/pixel_size # proxy for visium hires scale factor
                        ds_level = self.wsi.get_best_level_for_downsample(ds_factor)
                        tissue_hires_scalef = 1/self.wsi.level_downsamples()[ds_level]
                        
                    if TISSUE_LOWRES_SCALEF in scalefactors:
                        tissue_lowres_scalef = scalefactors[TISSUE_LOWRES_SCALEF]
                    if SPOT_DIAMETER_FULLRES in scalefactors:
                        spot_diameter_fullres_list.append(scalefactors[SPOT_DIAMETER_FULLRES])
                if IMAGES in keys:
                    image_data = self.adata.uns[SPATIAL][dataset_id][IMAGES]
                    if HIRES in image_data:
                        hires = image_data[HIRES]
                    else: 

                        # load wsi
                        def read_hest_wsi(wsi: WSI, width, height): 
                            return wsi.get_thumbnail(width, height)
        
                        if fullres: 
                            full_width, full_height = self.wsi.get_dimensions()
                            fullres = from_delayed(delayed(read_hest_wsi)(self.wsi, full_width, full_height), shape=(full_height, full_width, 3), dtype=np.int8)
                        else: 
                            fullres=None
                        hires_width, hires_height = self.wsi.level_dimensions()[ds_level]
                        hires = from_delayed(delayed(read_hest_wsi)(self.wsi, hires_width, hires_height), shape=(hires_height, hires_width, 3), dtype=np.int8)
                            
                    if LOWRES in image_data:
                        lowres = image_data[LOWRES]

                # construct the spatialdata elements
                if hires is not None:
                    # prepare the hires image
                    assert (
                        tissue_hires_scalef is not None
                    ), "tissue_hires_scalef is required when an the hires image is present"
                    hires = hires.transpose(2, 0, 1)
                    hires_image = Image2DModel.parse(
                        hires, 
                        dims=("c", "y", "x"),
                        transformations={f"{dataset_id}_downscaled_hires": Identity()}
                    )
                    hires_image = SpatialImage(hires_image, dims=("c", "y", "x"), name=f"{dataset_id}_downscaled_lowres_image")
                    images[f"{dataset_id}_downscaled_hires_image"] = hires_image

                    scale_hires = Scale([tissue_hires_scalef, tissue_hires_scalef], axes=("x", "y"))
                    shapes_transformations[f"{dataset_id}_downscaled_hires"] = scale_hires
                                    
                if fullres is not None: 
                    fullres = fullres.transpose(2, 0, 1)
                    
                    # compute scale factors: each scale level is relative to the previous level 
                    scale_factors = np.array([int(l) for l in self.wsi.level_downsamples()[1:] if full_height % l == 0 and full_width % l == 0])
                    scale_factors[1:] = scale_factors[1:] / scale_factors[:-1]
                    scale_factors = scale_factors.tolist()

                    fullres_image = Image2DModel.parse(
                        fullres, 
                        dims=("c", "y", "x"),
                        scale_factors=scale_factors,
                        transformations={f"{dataset_id}_fullres": Identity()}
                    )
                    images[f"{dataset_id}_fullres_image"] = fullres_image
                    scale_fullres = Scale([1, 1], axes=("x", "y"))
                    shapes_transformations[f"{dataset_id}_fullres"] = scale_fullres
                
                
                if lowres is not None:
                    assert (
                        tissue_lowres_scalef is not None
                    ), "tissue_lowres_scalef is required when an the lowres image is present"
                    lowres = lowres.transpose(2, 0, 1)
                    lowres_image = Image2DModel.parse(
                        lowres, dims=("c", "y", "x"), transformations={f"{dataset_id}_downscaled_lowres": Identity()}
                    )
                    lowres_image = SpatialImage(lowres_image, dims=("c", "y", "x"), name=f"{dataset_id}_downscaled_lowres")
                    images[f"{dataset_id}_downscaled_lowres_image"] = lowres_image

                    scale_lowres = Scale([tissue_lowres_scalef, tissue_lowres_scalef], axes=("x", "y"))
                    shapes_transformations[f"{dataset_id}_downscaled_lowres"] = scale_lowres

        # add cellvit and tissue contours
        for it in self.shapes: 
            shape = it.shapes
            key = it.name
            if len(shape) > 0 and isinstance(shape.iloc[0], Point): 
                shape['radius'] = 1
            val = ShapesModel.parse(shape, transformations=shapes_transformations)
            shapes[key] = val
        if self._tissue_contours is not None: 
            shapes['tissue_contours'] = ShapesModel.parse(self._tissue_contours, transformations=shapes_transformations)

        # validate the spot_diameter_fullres value
        if len(spot_diameter_fullres_list) > 0:
            d = np.array(spot_diameter_fullres_list)
            if not np.allclose(d, d[0]):
                warnings.warn(
                    "spot_diameter_fullres is not constant across datasets. Using the average value.",
                    UserWarning,
                    stacklevel=2,
                )
                spot_diameter_fullres = d.mean()
            else:
                spot_diameter_fullres = d[0]
        else:
            warnings.warn(
                f"spot_diameter_fullres is not present. Using {SPOT_DIAMETER_FULLRES_DEFAULT} as default value.",
                UserWarning,
                stacklevel=2,
            )
            spot_diameter_fullres = SPOT_DIAMETER_FULLRES_DEFAULT

        # parse and prepare the shapes
        if SPATIAL in self.adata.obsm:
            xy = self.adata.obsm[SPATIAL]
            radius = spot_diameter_fullres / 2
            shapes[REGION] = ShapesModel.parse(xy, geometry=0, radius=radius, transformations=shapes_transformations)

            # link the shapes to the table
            new_table = self.adata.copy()
            if TableModel.ATTRS_KEY in new_table.uns:
                del new_table.uns[TableModel.ATTRS_KEY]
            new_table.obs[REGION_KEY] = REGION
            new_table.obs[REGION_KEY] = new_table.obs[REGION_KEY].astype("category")
            new_table.obs[INSTANCE_KEY] = shapes[REGION].index.values
            new_table = TableModel.parse(new_table, region=REGION, region_key=REGION_KEY, instance_key=INSTANCE_KEY)
        else:
            new_table = self.adata.copy()
        
        return SpatialData(tables=new_table, images=images, shapes=shapes)
    
    def ensembl_id_to_gene(self):
        ensembl_id_to_gene(self)

    
class VisiumHESTData(HESTData): 
    def __init__(self, 
        adata: sc.AnnData, # type: ignore
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        tissue_seg: TissueMask=None,
        tissue_contours: gpd.GeoDataFrame=None,
        shapes: List[LazyShapes]=[]
    ):
        super().__init__(adata, img, pixel_size, meta, tissue_seg=tissue_seg, tissue_contours=tissue_contours, shapes=shapes)

class VisiumHDHESTData(HESTData): 
    def __init__(self, 
        adata: sc.AnnData, # type: ignore
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        tissue_seg: TissueMask=None,
        tissue_contours: gpd.GeoDataFrame=None,
        shapes: List[LazyShapes]=[]
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            shapes (List[LazyShapes]): dictionary of shapes, note that these shapes will be lazily loaded. Default: []
            tissue_seg (TissueMask): tissue mask for that sample
        """
        super().__init__(adata, img, pixel_size, meta, tissue_seg, tissue_contours, shapes)        
        
class STHESTData(HESTData):
    def __init__(self, 
        adata: sc.AnnData, # type: ignore
        img: Union[np.ndarray, str],
        pixel_size: float,
        meta: Dict = {},
        tissue_seg: TissueMask=None,
        tissue_contours: gpd.GeoDataFrame=None,
        shapes: List[LazyShapes]=[]
    ):
        """
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            img (Union[np.ndarray, str]): Full resolution image corresponding to the ST data, if passed as a path (str) the image is lazily loaded
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            tissue_seg (TissueMask): tissue mask for that sample
        """
        super().__init__(adata, img, pixel_size, meta, tissue_seg, tissue_contours, shapes)
        
class XeniumHESTData(HESTData):

    def __init__(
        self, 
        adata: sc.AnnData, # type: ignore
        img: Union[np.ndarray, openslide.OpenSlide, CuImage], # type: ignore
        pixel_size: float,
        meta: Dict = {},
        tissue_seg: TissueMask=None,
        tissue_contours: gpd.GeoDataFrame=None,
        shapes: List[LazyShapes]=[],
        xenium_nuc_seg: pd.DataFrame=None,
        xenium_cell_seg: pd.DataFrame=None,
        cell_adata: sc.AnnData=None, # type: ignore
        transcript_df: pd.DataFrame=None,
        dapi_path: str=None,
        alignment_file_path: str=None
    ):
        """
        class representing a single ST profile + its associated WSI image
        
        Args:
            adata (sc.AnnData): Spatial Transcriptomics data in a scanpy Anndata object (pooled by patch for Xenium)
                adata must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
            img (Union[np.ndarray, openslide.OpenSlide, CuImage]): Full resolution image corresponding to the ST data, Openslide/CuImage are lazily loaded, use CuImage for GPU accelerated computation
            pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
            meta (Dict): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
            shapes (List[LazyShapes]): dictionary of shapes, note that these shapes will be lazily loaded. Default: []
            tissue_seg (TissueMask): tissue mask for that sample
            xenium_nuc_seg (pd.DataFrame): content of a xenium nuclei contour file as a dataframe (nucleus_boundaries.parquet)
            xenium_cell_seg (pd.DataFrame): content of a xenium cell contour file as a dataframe (cell_boundaries.parquet)
            cell_adata (sc.AnnData): ST cell data, each row in adata.obs is a cell, each row in obsm is the cell location on the H&E image in pixels
            transcript_df (pd.DataFrame): dataframe of transcripts, each row is a transcript, he_x and he_y is the transcript location on the H&E image in pixels
            dapi_path (str): path to a dapi focus image
            alignment_file_path (np.ndarray): path to xenium alignment path
        """
        super().__init__(adata=adata, img=img, pixel_size=pixel_size, meta=meta, tissue_seg=tissue_seg, tissue_contours=tissue_contours, shapes=shapes)
        
        self.xenium_nuc_seg = xenium_nuc_seg
        self.xenium_cell_seg = xenium_cell_seg
        self.cell_adata = cell_adata
        self.transcript_df = transcript_df
        self.dapi_path = dapi_path
        self.alignment_file_path = alignment_file_path
        
        
    def save(
            self, 
            path: str, 
            save_img=True, 
            pyramidal=True, 
            bigtiff=False, 
            plot_pxl_size=False, 
            save_transcripts=False, 
            save_cell_seg=False, 
            save_nuclei_seg=False,
            **kwargs
        ):
        """Save a HESTData object to `path` as follows:
            - aligned_adata.h5ad (contains expressions for each spots + their location on the fullres image + a downscaled version of the fullres image)
            - metrics.json (contains useful metrics)
            - downscaled_fullres.jpeg (a downscaled version of the fullres image)
            - aligned_fullres_HE.tif (the full resolution image)
            - cells.geojson (cell segmentation if it exists)
            - Optional: cells_xenium.geojson (if xenium cell segmentation is attached to this object)
            - Optional: nuclei_xenium.geojson (if xenium cell segmentation is attached to this object)
            - Optional: tissue_contours.geojson (contours of the tissue segmentation if it exists)

        Args:
            path (str): save location
            save_img (bool): whenever to save the image at all (can save a lot of time if set to False)
            pyramidal (bool, optional): whenever to save the full resolution image as pyramidal (can be slow to save, however it's sometimes necessary for loading large images in QuPath). Defaults to True.
            bigtiff (bool, optional): whenever the bigtiff image is more than 4.1GB. Defaults to False.
        """
        super().save(path, save_img, pyramidal, bigtiff, plot_pxl_size)
        if self.cell_adata is not None:
            self.cell_adata.write_h5ad(os.path.join(path, 'aligned_cells.h5ad'))
        
        if save_transcripts and self.transcript_df is not None:
            self.transcript_df.to_parquet(os.path.join(path, 'aligned_transcripts.parquet'))

        if save_cell_seg:
            he_cells = self.get_shapes('tenx_cell', 'he').shapes
            he_cells.to_parquet(os.path.join(path, 'he_cell_seg.parquet'))
            write_geojson(he_cells, os.path.join(path, f'he_cell_seg.geojson'), '', chunk=True)
            
        if save_nuclei_seg:
            he_nuclei = self.get_shapes('tenx_nucleus', 'he').shapes
            he_nuclei.to_parquet(os.path.join(path, 'he_nucleus_seg.parquet'))
            write_geojson(he_nuclei, os.path.join(path, f'he_nucleus_seg.geojson'), '', chunk=True)
        

def read_HESTData(
    adata_path: str, 
    img: Union[str, np.ndarray, openslide.OpenSlide, CuImage],  # type: ignore
    metrics_path: str,
    mask_path_pkl: str = None, # Deprecated
    mask_path_jpg: str = None, # Deprecated
    cellvit_path: str = None,
    tissue_contours_path: str = None,
    xenium_cell_path: str = None,
    xenium_nucleus_path: str = None,
    transcripts_path: str = None
) -> HESTData:
    """ Read a HEST sample from disk

    Args:
        adata_path (str): path to .h5ad adata file containing ST data the 
            adata object must contain a downscaled image in ['spatial']['ST']['images']['downscaled_fullres']
        img (Union[str, np.ndarray, openslide.OpenSlide, CuImage]): path to a full resolution image (if passed as str) or full resolution image corresponding to the ST data, Openslide/CuImage are lazily loaded, use CuImage for GPU accelerated computation
        pixel_size (float): pixel_size of WSI im um/px, this pixel size will be used to perform operations on the slide, such as patching and segmenting
        metrics_path (str): metadata dictionary containing information such as the pixel size, or QC metrics attached to that sample
        mask_path_pkl (str): *Deprecated* path to a .pkl file containing the tissue segmentation contours. Defaults to None.
        mask_path_jpg (str): *Deprecated* path to a .jog file containing the greyscale tissue segmentation mask. Defaults to None.
        cellvit_path (str): path to a cell segmentation file in .geojson or .parquet. Defaults to None.
        tissue_contours_path (str): path to a .geojson tissue contours file. Defaults to None.
        xenium_cell_path (str): path to a .parquet xeniun cell segmentation file. Defaults to None.
        xenium_nucleus_path (str): path to a .parquet xenium nucleus segmentation file. Defaults to None.
        transcripts_path (str): path to a .parquet transcript dataframe. Defaults to None.


    Returns:
        HESTData: HESTData object
    """

    try:
        from cucim import CuImage
    except ImportError:
        CuImage = None
        CucimWarningSingleton.warn()

    import scanpy as sc

    if isinstance(img, str):
        if CuImage is not None:
            img = CuImage(img)
            width, height = img.resolutions['level_dimensions'][0]
        else:
            img = openslide.OpenSlide(img)
            width, height = img.dimensions
            
    tissue_contours = None
    tissue_seg = None
    if tissue_contours_path is not None:
        with open(tissue_contours_path) as f:
            lines = f.read()
            if 'hole' in lines:
                warnings.warn("this type of .geojson tissue contour file is deprecated, please download the new `tissue_seg` folder on huggingface: https://huggingface.co/datasets/MahmoodLab/hest/tree/main")
                gdf = TissueContourReader().read_gdf(tissue_contours_path)
                tissue_contours = old_geojson_to_new(gdf)
            else:
                tissue_contours = gpd.read_file(tissue_contours_path)
            
    elif mask_path_pkl is not None and mask_path_jpg is not None:
        tissue_seg = load_tissue_mask(mask_path_pkl, mask_path_jpg, width, height)
    
    shapes = []
    if cellvit_path is not None:
        shapes.append(LazyShapes(cellvit_path, 'cellvit', 'he'))
    if xenium_cell_path is not None:
        shapes.append(LazyShapes(xenium_cell_path, 'xenium_cell', 'he'))
    if xenium_nucleus_path is not None:
        shapes.append(LazyShapes(xenium_nucleus_path, 'xenium_nucleus', 'he'))
        
    transcripts = None
    if transcripts_path is not None:
        transcripts = pd.read_parquet(transcripts_path)
    
    adata = sc.read_h5ad(adata_path)
    with open(metrics_path) as metrics_f:     
        metrics = json.load(metrics_f)
        
    if transcripts is not None:
        return XeniumHESTData(
            adata, 
            img, 
            metrics['pixel_size_um_estimated'], 
            metrics, 
            tissue_seg=tissue_seg, 
            shapes=shapes, 
            tissue_contours=tissue_contours,
            transcript_df=transcripts
        )
    else:  
        return HESTData(
            adata, 
            img, 
            metrics['pixel_size_um_estimated'], 
            metrics, 
            tissue_seg=tissue_seg, 
            shapes=shapes, 
            tissue_contours=tissue_contours
        )
        

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
        

def create_benchmark_data(meta_df, save_dir:str, K):
    os.makedirs(save_dir, exist_ok=True)
    
    meta_df['patient'] = meta_df['patient'].fillna('Patient 1')
    
    get_k_genes_from_df(meta_df, 50, 'var', os.path.join(save_dir, 'var_50genes.json'))
    
    splits = meta_df.groupby(['dataset_title', 'patient'])['id'].agg(list).to_dict()
    create_splits(os.path.join(save_dir, 'splits'), splits, K=K)
    
    os.makedirs(os.path.join(save_dir, 'patches'), exist_ok=True)
    #mask_and_patchify_bench(meta_df, os.path.join(save_dir, 'patches'), use_mask=use_mask, keep_largest=keep_largest)
    
    os.makedirs(os.path.join(save_dir, 'patches_vis'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'adata'), exist_ok=True)
    for _, row in meta_df.iterrows():
        id = row['id']
        path = os.path.join(get_path_from_meta_row(row), 'processed')
        src_patch = os.path.join(path, 'patches.h5')
        dst_patch = os.path.join(save_dir, 'patches', id + '.h5')
        shutil.copy(src_patch, dst_patch)
        
        src_vis = os.path.join(path, 'patches_patch_vis.png')
        dst_vis = os.path.join(save_dir, 'patches_vis', id + '.png')
        shutil.copy(src_vis, dst_vis)
        
        src_adata = os.path.join(path, 'aligned_adata.h5ad')
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
        

class HESTIterator:
    def __init__(self, hest_dir, id_list, **read_kwargs):
        if id_list is not None and (not(isinstance(id_list, list) or isinstance(id_list, np.ndarray))):
            raise ValueError('id_list must a list or a numpy array')
        self.id_list = id_list
        self.hest_dir = hest_dir
        self.i = 0
        self.read_kwargs = read_kwargs
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self) -> HESTData:
        if self.i < len(self):
            x = _read_st(self.hest_dir, self.id_list[self.i], **self.read_kwargs)
            self.i += 1
            return x
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.id_list)

def iter_hest(hest_dir: str, id_list: List[str] = None, **read_kwargs) -> HESTIterator:
    """ Iterate through the HEST samples contained in `hest_dir`

    Args:
        hest_dir (str): hest directory containing folders: st, wsis, metadata, tissue_seg (optional)
        id_list (List[str], Optional): list of ids to read (ex: ['TENX96', 'TENX99']), pass None to read all available samples. Default to None
        load_transcripts (bool, Optional): Whenever to load the full transcript dataframe for Xenium samples. Default to False

    Returns:
        HESTIterator: HESTData iterator
    """
    return HESTIterator(hest_dir, id_list, **read_kwargs)

def _read_st(hest_dir, st_filename, load_transcripts=False):
    id = st_filename.split('.')[0]
    adata_path = os.path.join(hest_dir, 'st', f'{id}.h5ad')
    img_path = os.path.join(hest_dir, 'wsis', f'{id}.tif')
    meta_path = os.path.join(hest_dir, 'metadata', f'{id}.json')
    
    masks_path_pkl = None
    masks_path_jpg = None
    verify_paths([adata_path, img_path, meta_path], suffix='\nHave you downloaded the dataset? (https://huggingface.co/datasets/MahmoodLab/hest)')
    
    
    if os.path.exists(os.path.join(hest_dir, 'tissue_seg')):
        masks_path_pkl = find_first_file_endswith(os.path.join(hest_dir, 'tissue_seg'), f'{id}_mask.pkl')
        masks_path_jpg = find_first_file_endswith(os.path.join(hest_dir, 'tissue_seg'), f'{id}_mask.jpg')
        tissue_contours_path = find_first_file_endswith(os.path.join(hest_dir, 'tissue_seg'), f'{id}_contours.geojson')

    cellvit_path = None
    if os.path.exists(os.path.join(hest_dir, 'cellvit_seg')):
        cellvit_path = find_first_file_endswith(os.path.join(hest_dir, 'cellvit_seg'), f'{id}_cellvit_seg.parquet')
        if cellvit_path is None:
            cellvit_path = find_first_file_endswith(os.path.join(hest_dir, 'cellvit_seg'), f'{id}_cellvit_seg.geojson')
            if cellvit_path is not None:
                warnings.warn(f'reading the cell segmentation as .geojson can be slow, download the .parquet cells for faster loading https://huggingface.co/datasets/MahmoodLab/hest')
                
    if os.path.exists(os.path.join(hest_dir, 'xenium_seg')):
        xenium_cell_path = find_first_file_endswith(os.path.join(hest_dir, 'xenium_seg'), f'{id}_xenium_cell_seg.parquet')
        xenium_nucleus_path = find_first_file_endswith(os.path.join(hest_dir, 'xenium_seg'), f'{id}_xenium_nucleus_seg.parquet')
    else:
        xenium_cell_path = None
        xenium_nucleus_path = None
        
                    
    transcripts_path = None
    if load_transcripts:
        transcripts_path = find_first_file_endswith(os.path.join(hest_dir, 'transcripts'), f'{id}_transcripts.parquet')
                    
    st = read_HESTData(
        adata_path, 
        img_path, 
        meta_path, 
        masks_path_pkl, 
        masks_path_jpg, 
        cellvit_path=cellvit_path,
        tissue_contours_path=tissue_contours_path,
        xenium_cell_path=xenium_cell_path,
        xenium_nucleus_path=xenium_nucleus_path,
        transcripts_path=transcripts_path
    )
    return st
    
    

def load_hest(hest_dir: str, id_list: List[str] = None) -> List[HESTData]:
    """Read HEST-1k samples from a local directory

    Args:
        hest_dir (str): hest directory containing folders: st, wsis, metadata, tissue_seg (optional)
        id_list (List[str], Optional): list of ids to read (ex: ['TENX96', 'TENX99']). Default to None

    Returns:
        List[HESTData]: list of HESTData objects
    """
    
    if id_list is not None and (not(isinstance(id_list, list) or isinstance(id_list, np.ndarray))):
        raise ValueError('id_list must a list or a numpy array')
    
    warned = False
    
    hestdata_list = []
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    
    if id_list is not None:
        st_filenames = id_list
    else:
        st_filenames = os.listdir(os.path.join(hest_dir, 'st'))
        
    for st_filename in tqdm(st_filenames):
        st = _read_st(hest_dir, st_filename)
        hestdata_list.append(st)
        
    warnings.resetwarnings()
    return hestdata_list


def get_gene_db(species, cache_dir='.genes') -> pd.DataFrame:
    import scanpy as sc
    
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{species}.parquet')
    if not os.path.exists(path):
        print('querying biomart... (can take a few seconds)')
        annots = sc.queries.biomart_annotations(
            species,
            ["ensembl_gene_id", 'external_gene_name'],
        ).set_index('external_gene_name')
        annots.to_parquet(path)
    else:
        annots = pd.read_parquet(path)
        
    return annots


def _get_alias_to_parent_df():
    
    path_folder_assets = get_path_relative(__file__, '../../assets')
    path_gene_db = os.path.join(path_folder_assets, 'human_gene_db.parquet')
    
    if not os.path.exists(path_gene_db):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="MahmoodLab/hest", repo_type='dataset', local_dir=path_folder_assets, allow_patterns=['human_gene_db.parquet'])
    
    df = pd.read_parquet('assets/gene_db.parquet')
    df = df[['symbol', 'ensembl_gene_id', 'alias_symbol']].explode('alias_symbol')
    none_mask = df['alias_symbol'].isna()
    df.loc[none_mask, 'alias_symbol'] = df.loc[none_mask, 'symbol']
    df.index = df['alias_symbol']
    df = df[~df.index.duplicated('first')]
    return df

def unify_gene_names(adata: sc.AnnData, species="human", drop=False) -> sc.AnnData: # type: ignore
    """ unify gene names by resolving aliases

    Args:
        adata (sc.AnnData): scanpy anndata
        species (str, optional): species, choose between ["human", "mouse"]. Defaults to "human".
        drop (bool, optional): whenever to drop gene names having no alias. Defaults to False.

    Returns:
        sc.AnnData: anndata with unified gene names in var_names
    """
    adata = adata.copy()
    duplicated_genes_before = adata.var_names[adata.var_names.duplicated()]

    var_names = adata.var_names.values


    alias_to_parent_df = _get_alias_to_parent_df()
    parent_names = alias_to_parent_df['symbol'].values
    
    var_names = np.unique(var_names)
    
    # Conventional gene names in adata
    adata_conv_genes = np.intersect1d(parent_names, var_names)
    
    # Unconventional (alias) gene names in adata
    unknown_genes = np.setdiff1d(var_names, adata_conv_genes, assume_unique=True)
    logger.info(f'Found {len(unknown_genes)} unknown genes out of {len(var_names)}')
    
    parent_names = alias_to_parent_df.reindex(unknown_genes)
    
    matched_parent = parent_names.dropna()
    
    # If the parent of the alias already exists, keep the alis
    parent_already_exists = matched_parent['symbol'].isin(var_names)
    matched_parent = matched_parent[~parent_already_exists]
    
    remaining = parent_names.drop(matched_parent.index, axis=0)
    
    logger.info(f"Mapped {len(matched_parent)} aliases to their parent name, {len(remaining)} remaining unknown genes")
    
    mapped = pd.DataFrame(var_names.copy(), index=var_names)
    mapped.loc[matched_parent.index, 0] = matched_parent['symbol'].values
    
    mask = mapped.isna()[0].values
    mapped.loc[mask, 0] = mapped.index[mask].values
    adata.var_names =  mapped.loc[adata.var_names][0].values
    
    duplicated_genes_after = adata.var_names[adata.var_names.duplicated()]
    if len(duplicated_genes_after) > len(duplicated_genes_before):
        logger.warning(f"duplicated genes increased from {len(duplicated_genes_before)} to {len(duplicated_genes_after)} after resolving aliases")
        logger.warning('deduplicating... (can remove useful genes)')
        mask = ~adata.var_names.duplicated(keep='first')
        adata = adata[:, mask]
    
    duplicated_genes_after = adata.var_names[adata.var_names.duplicated()]
    if len(duplicated_genes_after) > len(duplicated_genes_before):
        logger.warning(f"duplicated genes increased from {len(duplicated_genes_before)} to {len(duplicated_genes_after)} after resolving aliases")
    logger.info('deduplicating...')
    mask = ~adata.var_names.duplicated(keep='first')
    adata = adata[:, mask]
    
    if drop:
        adata = adata[:, ~remaining]
 
    # TODO return dict map of renamed, and remaining
    return adata

def ensembl_id_to_gene(st: HESTData, filter_na = False) -> HESTData:
    """
    Converts ensemble gene IDs of a HESTData object using Biomart annotations and filter out genes with no matching Ensembl ID
    
    Args: 
        st (HESTData): HESTData object
        filter_na (bool): whenever to filter genes that are not valid ensemble IDs. Defaults to False.
    
    Returns: 
        HESTData: HESTData object with gene names instead of ensemble gene IDs
    """

    import scanpy as sc
    species = st.meta['species']
    org = "hsapiens" if species == "Homo sapiens" else "mmusculus"
    
    annotations = sc.queries.biomart_annotations(org=org,attrs=['ensembl_gene_id', 'external_gene_name'], use_cache=True)
    ensembl_to_gene_name = dict(zip(annotations['ensembl_gene_id'], annotations['external_gene_name']))

        
    st.adata.var['gene_name'] = st.adata.var_names.map(ensembl_to_gene_name, na_action=None)
    
    if filter_na: 
        st.adata.var_names = st.adata.var['gene_name'].fillna('')
    else: 
        st.adata.var['gene_name'] = st.adata.var['gene_name'].where(st.adata.var['gene_name'].notna(), st.adata.var_names)
        
    valid_genes = st.adata.var['gene_name'].notna()
    st.adata = st.adata[:, valid_genes]


    return st


def save_spatial_plot(adata: sc.AnnData, save_path: str, name: str='', key='total_counts', pl_kwargs={}):
    """Save the spatial plot from that sc.AnnData

    Args:
        save_path (str): path to a directory where the spatial plot will be saved
        name (str): save plot as {name}spatial_plots.png
        key (str): feature to plot. Default: 'total_counts'
        pl_kwargs(Dict): arguments for sc.pl.spatial
    """
    import scanpy as sc
    
    fig = sc.pl.spatial(adata, show=False, img_key="downscaled_fullres", color=[key], title=f"in_tissue spots", return_fig=True, **pl_kwargs)
    
    filename = f"{name}spatial_plots.png"
    
    # Save the figure
    fig.savefig(os.path.join(save_path, filename), dpi=400)