import openslide
import numpy as np
import cv2
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import math
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import os
from .vst_save_utils import initsave_hdf5
import pickle

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

class WSI:
    wsi: openslide.OpenSlide = None
    contours_holes = None
    contours_tissue = None
    
    def save_segmentation(self, mask_file):
        asset_dict = self.get_segmentation()
        save_pkl(mask_file, asset_dict)


    def get_segmentation(self):
        asset_dict = {'holes': self.contours_holes, 
                      'tissue': self.contours_tissue, 
                      'groups': None}
        return asset_dict


    def visualize_wsi(
                self,
                vis_level=-1,
                line_color=(0, 255, 0),
                hole_color=(0, 0, 255),
                annot_color=(255, 0, 0),
                line_thickness=250,
                downscale=64,
                view_slide_only=False,
                seg_display=True,
                annot_display=True,
                show_group=False,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_size=2,
                font_thickness=10,
                cont_df=None
        ):
            if vis_level == -1:
                vis_level = self.wsi.get_best_level_for_downsample(downscale)

            top_left = (0,0)
            downsample = self.wsi.level_downsamples[vis_level]
            scale = [1/downsample, 1/downsample]    
            region_size = self.wsi.level_dimensions[vis_level]
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
            
            if view_slide_only:
                return Image.fromarray(img)

            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            draw_cont = partial(cv2.drawContours, contourIdx=-1, thickness=line_thickness, lineType=cv2.LINE_8, offset=offset)
            put_text = partial(cv2.putText, fontFace=font, fontScale=font_size, thickness=font_thickness)

            if self.contours_tissue is not None and seg_display:
                for idx, cont in enumerate(self.contours_tissue):
                    cont = np.array(scale_contour_dim(cont, scale))
                    M = cv2.moments(cont)
                    ##cX = int(M["m10"] / M["m00"]) # warning: can be zero
                    #cY = int(M["m01"] / M["m00"]) # warning: can be zero
                    draw_cont(image=img, contours=[cont], color=line_color)

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

            return Image.fromarray(img)


    def dump_patches(
        self,
        patch_save_dir: str,
        adata: sc.AnnData, 
        src_pixel_size: float,
        name: str = None,
        patch_size_um: float=128,
        tissue_mask: np.ndarray = None,
        target_pixel_size: float=0.5,
        verbose=0
    ):

        #TODO change
        #img = self.wsi.read_region((0, 0), 0, self.wsi.dimensions)
        #img = np.array(img)
        #if img.shape[2] == 4:
        #    img = img[:, :, :3]
        
        # minimum intersection percecentage with the tissue mask to keep a patch
        TISSUE_INTER_THRESH = 0.05
        
        scale_factor = src_pixel_size / target_pixel_size
        patch_size_pxl = round(patch_size_um / src_pixel_size)
        patch_count = 0
        output_datafile = os.path.join(patch_save_dir, name + '.h5')

        assert len(adata.obs) == len(adata.obsm['spatial'])
        
        mode_HE = 'w'
        i = 0
        img_width, img_height = self.wsi.dimensions
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
            
            #image_patch = img[yImage - patch_size_pxl // 2: yImage + patch_size_pxl // 2,
            #                    xImage - patch_size_pxl // 2: xImage + patch_size_pxl // 2, :]
            image_patch = self.wsi.read_region((xImage - patch_size_pxl // 2, yImage - patch_size_pxl // 2), 0, (patch_size_pxl, patch_size_pxl))
            image_patch = np.array(image_patch)
            if image_patch.shape[2] == 4:
                image_patch = image_patch[:, :, :3]
            
            if tissue_mask is not None:
                patch_mask = tissue_mask[yImage - patch_size_pxl // 2: yImage + patch_size_pxl // 2,
                                xImage - patch_size_pxl // 2: xImage + patch_size_pxl // 2]
                patch_area = patch_mask.shape[0] ** 2
                pixel_count = patch_mask.sum()

                if pixel_count / patch_area < TISSUE_INTER_THRESH:
                    continue
            
            patch_count += 1
            image_patch = cv2.resize(image_patch, (round(scale_factor * patch_size_pxl), round(scale_factor * patch_size_pxl)), interpolation=cv2.INTER_CUBIC)
            
            #image = Image.fromarray(image_patch)
            #image.save(f'/mnt/sdb1/paul/test_patch/{barcode_spot}.png')
            
            # Save ref patches

            asset_dict = { 'img': np.expand_dims(image_patch, axis=0),  # (1 x w x h x 3)
                            'coords': np.expand_dims([yImage, xImage], axis=0),   # (1 x 2)
                            'barcode': np.expand_dims([barcode_spot], axis=0)
                            }

            
        
            attr_dict = {}
            attr_dict['img'] = {'patch_size': patch_size_pxl,
                                'factor': scale_factor}

            initsave_hdf5(output_datafile, asset_dict, attr_dict, mode=mode_HE, verbose=1)
            mode_HE = 'a'

            if patch_count < 10:
                my_img = Image.fromarray(image_patch)
                my_img.save(os.path.join('/media/ssd2/hest/patch_samples', name + '_' + str(patch_count) +  '.png'))

            # Save both raw and smoothed version of spot gene expression
            #output_datafile = os.path.join(gene_save_dir, name + '.h5')
            #gene_exp_raw = gene_matrix_pp[index0, :]
            #gene_exp_smooth = smooth_gene_exp(barcode_spot, df_alphabetic, df_in_tissue, gene_matrix_pp) 


            #asset_dict = { 'raw': np.expand_dims(gene_exp_raw, axis=0),  # (1 x NumOfGenes)
            #                'smooth': np.expand_dims(gene_exp_smooth, axis=0),  # (1 x NumOfGenes)
            #                'coords': np.expand_dims([yImage, xImage], axis=0),   # (1 x 3)
            #                'barcode': np.expand_dims([barcode_spot], axis=0)
            #                }

            #initsave_hdf5(output_datafile, asset_dict, mode=mode_gene)
            #mode_gene = 'a'
            
        if verbose:
            print(f'found {patch_count} valid patches')


def scale_contour_dim(contours, scale):
    r"""
    """
    return [np.array(cont * scale, dtype='int32') for cont in contours]


def scale_holes_dim(contours, scale):
    r"""
    """
    return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]


def mask_and_patchify(meta_df: pd.DataFrame):
    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        #path = _get_path_from_meta_row(row)
        id = row['id']
        img_path = f'/media/ssd2/hest/pyramidal/{id}.tif'
        mask_path = f'/media/ssd2/hest/masks/{id}_mask.npy'
        adata_path = f'/media/ssd2/hest/adata/{id}.h5ad'
        adata = sc.read_h5ad(adata_path)
        pixel_size = row['pixel_size_um_estimated']
        mask = np.load(mask_path)
        #mask = np.transpose(mask, (1, 0))
        wsi = WSI()
        wsi.wsi = openslide.OpenSlide(img_path)

        mask_to_contours(wsi, mask)


        image_vis = wsi.visualize_wsi(line_thickness=5)
        plt.imshow(image_vis)
        plt.savefig(f'/media/ssd2/hest/masks_vis/{id}_vis.png')
        wsi.save_segmentation(f'/media/ssd2/hest/masks_pkl/{id}_mask.pkl')

        wsi.dump_patches('/media/ssd2/hest/patches',
                           adata,
                           pixel_size,
                           id,
                           tissue_mask=mask,
                           verbose=1)
    


def filter_contours(wsi, seg_level, contours, hierarchy, filter_params, scale):
    """
        Filter contours by: area
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    if len(hierarchy) == 0:
        hierarchy_1 = []
    else:
        hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
    all_holes = []
    
    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()

        if a == 0: continue

        
        
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            
            if (filter_params['filter_color_mode'] == 'none') or (filter_params['filter_color_mode'] is None):
                filtered.append(cont_idx)
                all_holes.append(holes)
            else:
                scale = wsi.level_downsamples[seg_level]
                orig_cont = np.array(cont * scale, dtype='int32')
                x, y, w, h = cv2.boundingRect(orig_cont)
                patch = np.array(wsi.read_region((x,y), seg_level, (int(w/scale[0]),int(h/scale[1]))))
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
                is_not_dirty = filter_color(hsv_patch, a, filter_params['filter_color_mode'])
                if is_not_dirty:
                    filtered.append(cont_idx)
                    all_holes.append(holes)

    
    # for parent in filtered:
    # 	all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

    ##### TODO: re-implement this in a single for-loop that 
    ##### loops through both parent contours and holes

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]
    
    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids ]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []
        
        # filter these holes
        #for hole in unfilered_holes:
        #    if cv2.contourArea(hole) > filter_params['a_h']:
        #        filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours
        
def mask_to_contours(wsi: openslide.OpenSlide, mask: np.ndarray, keep_ids = [], exclude_ids=[]):
    TARGET_EDGE_SIZE = 2000
    scale = TARGET_EDGE_SIZE / mask.shape[0]

    downscaled_mask = cv2.resize(mask, (round(mask.shape[0] * scale), round(mask.shape[1] * scale)))

     # Find and filter contours
    contours, hierarchy = cv2.findContours(downscaled_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
    #print('Num Contours Before Filtering:', len(contours))
    if hierarchy is None:
        hierarchy = []
    else:
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    filter_params = {
        'filter_color_mode': 'none',
        'max_n_holes': 20,
        'a_t': 1000
    }

    seg_level = 0
    if filter_params: 
        foreground_contours, hole_contours = filter_contours(wsi, seg_level, contours, hierarchy, filter_params, scale)  # Necessary for filtering out artifacts

    
    if len(foreground_contours) == 0:
        raise Exception('no contour detected')
    else:
        wsi.contours_tissue = scale_contour_dim(foreground_contours, 1 / scale)
        wsi.contours_holes = scale_holes_dim(hole_contours, 1 / scale)

    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        contour_ids = set(np.arange(len(wsi.contours_tissue))) - set(exclude_ids)

    wsi.contours_tissue = [wsi.contours_tissue[i] for i in contour_ids]
    wsi.contours_holes = [wsi.contours_holes[i] for i in contour_ids]

    #print('Num Contours After Filtering:', len(wsi.contours_tissue))


    #wsi.seg_level = seg_level


    
def mask_spots(
    adata: sc.AnnData,
    src_pixel_size: float,
    tissue_mask: np.ndarray = None,
    spot_diameter_um: float = 55.
) -> sc.AnnData:
    
    if not(0 <= tissue_mask.min() and tissue_mask.max() <= 1):
        raise ValueError("tissue_mask values aren't between 0 and 1")
    
    filtered_adata = adata.copy()
    
    spot_dia_pxl = spot_diameter_um * src_pixel_size
    spot_rad_pxl = round(spot_dia_pxl / 2)
    
    for _, row in tqdm(adata.obs.iterrows(), total=len(adata.obs)):
        
        barcode_spot = row.index

        # Extract H&E patch
        xImage = row['pxl_col_in_fullres']
        yImage = row['pxl_row_in_fullres']
    
        mask_patch = tissue_mask[yImage - spot_rad_pxl: yImage + spot_rad_pxl,
                            xImage - spot_rad_pxl: xImage + spot_rad_pxl, :]
        
        circle_patch = np.zeros(mask_patch.shape)
        
        #mask_patch_area = patch_size_pxl ** 2
        #if (mask_patch.sum() / mask_patch_area) < 0.95:
        #    continue
        
        x, y = np.meshgrid(np.arange(spot_rad_pxl), np.arange(spot_rad_pxl))

        # Calculate the distances from each point to the circle's center
        distances = np.sqrt((x - spot_rad_pxl)**2 + (y - spot_rad_pxl)**2)

        # Create a mask for the circle's region using the distances and radius
        circle_region_mask = distances <= spot_rad_pxl

        # Set the circle's region in the image array to 1
        circle_patch[circle_region_mask] = 1
        spot_area_pxl = np.pi * spot_rad_pxl**2
        inter_pct = (circle_patch & mask_patch) / spot_area_pxl
        if inter_pct < 0.95:
            print(f'filter out barcode {barcode_spot}')
            filtered_adata = filtered_adata[filtered_adata.obs_names != barcode_spot]
            
    return filtered_adata
    
    
