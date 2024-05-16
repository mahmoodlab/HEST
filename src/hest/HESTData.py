import json
import os
import shutil
from functools import partial
from typing import Dict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
import scanpy as sc
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from matplotlib import rcParams
from matplotlib.collections import PatchCollection
from PIL import Image
from tqdm import tqdm

from src.hest.masking import (apply_otsu_thresholding, keep_largest_area,
                              mask_to_contours, save_pkl, scale_contour_dim)
from src.hest.utils import (ALIGNED_HE_FILENAME, get_path_from_meta_row,
                            plot_verify_pixel_size, save_scalefactors,
                            tiff_save, write_10X_h5)

from .vst_save_utils import initsave_hdf5


class HESTData:
    """
    Object representing a single Spatial Transcriptomics sample along with a full resolution H&E image and metadatas
    """
    h5_path = None
    spatial_path = None
    save_positions = True
    tissue_mask = None
    thumbnail = None
    contours_tissue = None
    
    
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
        
        for index in adata.obs.index:
            if len(index) != len(adata.obs.index[0]):
                raise Exception('indices of adata.obs must all have the same length, otherwise problems can occur when saving to h5')
    
    
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
                and the following collomns in adata.obs: ['array_col', 'array_row', 'in_tissue', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
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
            save_path (str): path to a directory where the spatial plot will be saved
        """
        print("Plotting spatial plots...")
             
        sc.pl.spatial(self.adata, show=None, img_key="downscaled_fullres", color=['total_counts'], title=f"in_tissue spots", alpha=0.4)
        
        filename = f"spatial_plots.png"
        
        # Save the figure
        plt.savefig(os.path.join(save_path, filename))
        plt.close()  # Close the plot to free memory
        print(f"H&E overlay spatial plots saved in {save_path}")
    
        
    def save(self, path: str, pyramidal=True):
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
            save_scalefactors(self.adata, os.path.join(path, 'spatial/scalefactors_json.json'))

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
        
        
        plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, os.path.join(path, 'pixel_size_vis.png'))
        
        pixel_size = self.meta['pixel_size_um_estimated']
        
        tiff_save(self.img, os.path.join(path, ALIGNED_HE_FILENAME), pixel_size, pyramidal=pyramidal)
        
        
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
        

    def visualize_mask_and_patches(
                self,
                vis_level=-1,
                line_color=(0, 255, 0),
                hole_color=(0, 0, 255),
                annot_color=(255, 0, 0),
                line_thickness=250,
                target_width=1000,
                view_slide_only=False,
                seg_display=True,
                annot_display=True,
                show_group=False,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_size=2,
                font_thickness=10,
                cont_df=None
        ):
            #if vis_level == -1:
            #    vis_level = self.wsi.get_best_level_for_downsample(downscale)
            #width, height = self.wsi.dimensions
            height, width = self.img.shape[:2]
            downsample = target_width / width

            top_left = (0,0)
            #downsample = self.wsi.level_downsamples[vis_level]
            scale = [downsample, downsample]    
            #region_size = self.wsi.level_dimensions[vis_level]
            #img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
            #cv2.resize(self.downscaled_img)
            img = np.array(cv2.resize(self.img, dsize=(round(width * downsample), round(height * downsample))))
            #img = np.array(self.wsi.get_thumbnail((width * downsample, height * downsample)))
            self.downscaled_img = img.copy()


            downscaled_mask = cv2.resize(self.tissue_mask, (img.shape[1], img.shape[0]))
            downscaled_mask = np.expand_dims(downscaled_mask, axis=-1)
            downscaled_mask = downscaled_mask * np.array([0, 0, 0]).astype(np.uint8)

            if view_slide_only:
                return Image.fromarray(img)

            offset = tuple(-(np.array(top_left) * scale).astype(int))
            #line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            draw_cont = partial(cv2.drawContours, contourIdx=-1, thickness=line_thickness, lineType=cv2.LINE_8, offset=offset)
            draw_cont_fill = partial(cv2.drawContours, contourIdx=-1, thickness=cv2.FILLED, offset=offset)
            put_text = partial(cv2.putText, fontFace=font, fontScale=font_size, thickness=font_thickness)

            if self.contours_tissue is not None and seg_display:
                for idx, cont in enumerate(self.contours_tissue):
                    cont = np.array(scale_contour_dim(cont, scale))
                    M = cv2.moments(cont)
                    ##cX = int(M["m10"] / M["m00"]) # warning: can be zero
                    #cY = int(M["m01"] / M["m00"]) # warning: can be zero
                    draw_cont(image=img, contours=[cont], color=line_color)
                    draw_cont_fill(image=downscaled_mask, contours=[cont], color=line_color)

                    if cont_df is not None:
                        if idx not in cont_df.index: 
                            continue
                        label = str(cont_df.loc[idx, 'label'])
                    else:
                        label = str(idx)

                    #put_text(img=img, text=label, org=(cX, cY), color=(255, 0, 0))
                    #if show_group:
                    #    put_text(img=img, text=str(wsi.groups[idx]), org=(cX+20, cY+20), color=(0, 0, 255))

                ### Draw hole contours
                for cont in self.contours_holes:
                    cont = scale_contour_dim(cont, scale)
                    draw_cont(image=img, contours=cont, color=hole_color) 

            alpha = 0.4
            self.downscaled_mask = downscaled_mask
            #img = cv2.drawContours(img, self.contours_tissue, contourIdx=-1, thickness=cv2.FILLED, offset=offset, color=(255, 255, 255))
            #overlay_array = (downscaled_mask * [144, 238, 144]).astype(np.uint8)
            self.tissue_mask = cv2.resize(downscaled_mask, self.tissue_mask.shape).round().astype(np.uint8)
            img = cv2.addWeighted(img, 1 - alpha, downscaled_mask, alpha, 0)
            img = img.astype(np.uint8)

            return Image.fromarray(img)


    def _compute_mask(self, keep_largest=False):
        #width, height = self.wsi.dimensions
        height, width = self.img.shape[:2]
        TARGET_WIDTH = 2000
        scale = TARGET_WIDTH / width
        thumbnail = cv2.resize(self.img, (round(height * scale), round(width * scale)))
        #thumbnail = np.array(self.wsi.get_thumbnail((width * scale, height * scale)))
        #Image.fromarray(thumbnail).save('thumb.png')
        mask = apply_otsu_thresholding(thumbnail).astype(np.uint8)
        mask = 1 - mask
        if keep_largest:
            mask = keep_largest_area(mask)
        self.tissue_mask = np.round(cv2.resize(mask, (height, width))).astype(np.uint8)
        self.contours_tissue, self.contours_holes = mask_to_contours(self.tissue_mask)


    def dump_patches(
        self,
        patch_save_dir: str,
        adata: sc.AnnData, 
        src_pixel_size: float,
        name: str = None,
        target_patch_size: int=224,
        #patch_size_um: float=112,
        target_pixel_size: float=0.5,
        verbose=0,
        dump_visualization=True,
        use_mask=True,
        load_in_memory=True,
        keep_largest=False
    ):

        #TODO change
        #img = self.wsi.read_region((0, 0), 0, self.wsi.dimensions)
        #img = np.array(img)
        #if img.shape[2] == 4:
        #    img = img[:, :, :3]
        
        # minimum intersection percecentage with the tissue mask to keep a patch
        TISSUE_INTER_THRESH = 0.05
        TARGET_VIS_SIZE = 1000
        
        scale_factor = target_pixel_size / src_pixel_size
        #patch_size_pxl = round(patch_size_um  src_pixel_size)
        patch_size_pxl = round(target_patch_size * scale_factor)
        patch_count = 0
        output_datafile = os.path.join(patch_save_dir, name + '.h5')

        assert len(adata.obs) == len(adata.obsm['spatial'])

        fig, ax = plt.subplots()
        
        mode_HE = 'w'
        i = 0
        if load_in_memory:
            img_height, img_width = self.img.shape[:2]
        else:
            raise NotImplementedError()
            img_width, img_height = self.wsi.dimensions
        patch_rectangles = [] # lower corner (x, y) + (widht, height)
        downscale_vis = TARGET_VIS_SIZE / img_width

        if self.tissue_mask is None and use_mask:
            self._compute_mask(keep_largest)
        elif not use_mask:
            self.tissue_mask = np.ones((img_height, img_width)).astype(np.uint8)

        mask_plot = self.visualize_mask_and_patches(line_thickness=3, target_width=1000)

        ax.imshow(mask_plot)
        #ax.imshow(self.wsi.get_thumbnail((img_width * downscale_vis, img_height * downscale_vis)))
        for index0, row in tqdm(adata.obs.iterrows(), total=len(adata.obs)):
            
            barcode_spot = row.name

            xImage = int(adata.obsm['spatial'][i][0]) #int(row['pxl_col_in_fullres'])
            yImage = int(adata.obsm['spatial'][i][1]) #int(row['pxl_row_in_fullres'])

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
                raise NotImplementedError()
                image_patch = self.wsi.read_region((xImage - patch_size_pxl // 2, yImage - patch_size_pxl // 2), 0, (patch_size_pxl, patch_size_pxl))
            rect_x = (xImage - patch_size_pxl // 2) * downscale_vis
            rect_y = (yImage - patch_size_pxl // 2) * downscale_vis
            rect_width = patch_size_pxl * downscale_vis
            rect_height = patch_size_pxl * downscale_vis

            image_patch = np.array(image_patch)
            if image_patch.shape[2] == 4:
                image_patch = image_patch[:, :, :3]
                
            
            if use_mask:
                patch_mask = self.tissue_mask[yImage - patch_size_pxl // 2: yImage + patch_size_pxl // 2,
                                xImage - patch_size_pxl // 2: xImage + patch_size_pxl // 2]
                patch_area = patch_mask.shape[0] ** 2
                pixel_count = patch_mask.sum()

                if pixel_count / patch_area < TISSUE_INTER_THRESH:
                    continue

            patch_rectangles.append(matplotlib.patches.Rectangle((rect_x, rect_y), rect_width, rect_height))
            
            patch_count += 1
            image_patch = cv2.resize(image_patch, (target_patch_size, target_patch_size), interpolation=cv2.INTER_CUBIC)
            
            #image = Image.fromarray(image_patch)
            #image.save(f'/mnt/sdb1/paul/test_patch/{barcode_spot}.png')
            
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
            

    def get_segmentation(self):
        asset_dict = {'holes': self.contours_holes, 
                      'tissue': self.contours_tissue, 
                      'groups': None}
        return asset_dict
      
            
    def save_segmentation(self, save_dir, name, deeplab=False):
        if self.tissue_mask is None:
            self._compute_mask()

        image_vis = self.visualize_wsi(line_thickness=3)

        # save to a deeplab compatible format
        if deeplab:
            #TARGET_WIDTH = 1000
            #scale = TARGET_WIDTH / self.tissue_mask.shape[0]
            #tissue_mask = cv2.resize(self.tissue_mask, dsize=(int(scale * self.tissue_mask.shape[0]), int(scale * self.tissue_mask.shape[1])))
            #tissue_mask = np.expand_dims(tissue_mask, axis=-1)
            #tissue_mask = tissue_mask * np.array([255, 255, 255]).astype(np.uint8)
            #width, height = self.wsi.dimensions
            #downscaled_img = self.wsi.get_thumbnail((int(width * scale), int(height * scale)))
            os.makedirs(os.path.join(save_dir, 'deeplab', 'Masks'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'deeplab', 'Images'), exist_ok=True)
            Image.fromarray(self.downscaled_mask).save(os.path.join(save_dir, 'deeplab', 'mask', f'{name}.png'))

            Image.fromarray(self.downscaled_img).save(os.path.join(save_dir, 'deeplab', 'image', f'{name}.png'))


        os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
        image_vis.save(os.path.join(save_dir, 'vis', f'{name}_vis.png'))
        asset_dict = self.get_segmentation()
        save_pkl(os.path.join(save_dir, f'{name}_mask.pkl'), asset_dict)


class VisiumHESTData(HESTData): 
    def __init__(self, adata: sc.AnnData, img: np.ndarray, meta: Dict):
        super().__init__(adata, img, meta, spot_size=55., spot_inter_dist=100.)

class VisiumHDHESTData(HESTData): 
    def __init__(self, adata: sc.AnnData, img: np.ndarray, meta: Dict):
        super().__init__(adata, img, meta, spot_size=128., spot_inter_dist=128.)        
        
class STHESTData(HESTData):
    def __init__(self, adata: sc.AnnData, img: np.ndarray, meta: Dict):
        super().__init__(adata, img, meta, spot_size=100., spot_inter_dist=200.)
        self.save_positions = False
        
class XeniumHESTData(HESTData):
    def __init__(self, adata: sc.AnnData, img: np.ndarray, meta: Dict):
        super().__init__(adata, img, meta, spot_size=55., spot_inter_dist=100.)
        self.save_positions = False


def read_HESTData(adata_path: str, pyramidal_tiff_path: str, metrics_path: str) -> HESTData:
    adata = sc.read_h5ad(adata_path)
    image = pyvips.Image.tiffload(pyramidal_tiff_path).numpy()
    with open(metrics_path) as metrics_f:     
        metrics = json.load(metrics_f)
    return HESTData(adata, image, metrics, spot_inter_dist=metrics['inter_spot_dist'], spot_size=metrics['spot_diameter'])
        

def mask_and_patchify(meta_df: pd.DataFrame, save_dir: str, use_mask=True, keep_largest=None):
    i = 0
    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        id = row['id']
        img_path = f'/mnt/sdb1/paul/images/pyramidal/{id}.tif'
        adata_path = f'/mnt/sdb1/paul/images/adata/{id}.h5ad'
        adata = sc.read_h5ad(adata_path)
        #pixel_size = row['pixel_size_um_estimated']
        metrics_path = os.path.join(get_path_from_meta_row(row), 'processed', 'metrics.json')
        
        #mask = np.load(mask_path)
        #mask = np.transpose(mask, (1, 0))
        hest_obj = read_HESTData(adata_path, img_path, metrics_path)
        #wsi = WSI(img_path)


        keep_largest_args = keep_largest[i] if keep_largest is not None else False

        hest_obj.dump_patches(save_dir,
                           adata,
                           hest_obj.meta['pixel_size_um_estimated'],
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
    mask_and_patchify(meta_df, os.path.join(save_dir, 'patches'), use_mask=use_mask, keep_largest=keep_largest)
    
    os.makedirs(os.path.join(save_dir, 'adata'), exist_ok=True)
    for index, row in meta_df.iterrows():
        id = row['id']
        src_adata = os.path.join(adata_folder, id + '.h5ad')
        dst_adata = os.path.join(save_dir, 'adata', id + '.h5ad')
        shutil.copy(src_adata, dst_adata)
        
        
def create_splits(dest_dir, splits, K):
    # [[patien1], [patien2]]...
        

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
            
            
    arr = [value for key, value in splits.items()]
    split_nb = 0
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
        #train_ids, test_ids = splits[k]

        data_train = np.column_stack((train_ids, [os.path.join('patches', id + '.h5') for id in train_ids], [os.path.join('adata', id + '.h5ad') for id in train_ids]))
        train_df = pd.DataFrame(data_train, columns=['sample_id', 'patches_path', 'expr_path'])

        data_test = np.column_stack((test_ids, [os.path.join('patches', id + '.h5') for id in test_ids], [os.path.join('adata', id + '.h5ad') for id in test_ids]))
        test_df = pd.DataFrame(data_test, columns=['sample_id', 'patches_path', 'expr_path'])
        train_df.to_csv(os.path.join(dest_dir, f'train_{i}.csv'), index=False)
        test_df.to_csv(os.path.join(dest_dir, f'test_{i}.csv'), index=False)
        
