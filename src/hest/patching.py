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
from matplotlib.collections import PatchCollection
import matplotlib
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()


def mask_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask an RGB image

    Args:
        rgb (np.ndarray): RGB image to mask with shape (height, width, 3)
        mask (np.ndarray): Binary mask with shape (height, width)

    Returns:
        np.ndarray: Masked image
    """
    assert (
        rgb.shape[:-1] == mask.shape
    ), "Mask and RGB shape are different. Cannot mask when source and mask have different dimension."
    mask_positive = np.dstack([mask, mask, mask])
    mask_negative = np.dstack([~mask, ~mask, ~mask])
    positive = rgb * mask_positive
    negative = rgb * mask_negative
    negative = 255 * (negative > 0.0001).astype(int)

    masked_image = positive + negative

    return np.clip(masked_image, a_min=0, a_max=255)


def apply_otsu_thresholding(tile: np.ndarray) -> np.ndarray:
    """Generate a binary tissue mask by using Otsu thresholding

    Args:
        tile (np.ndarray): Tile with tissue with shape (height, width, 3)

    Returns:
        np.ndarray: Binary mask with shape (height, width)
    """

    # this is to remove the black border padding in some images
    black_pixels = np.all(tile == [0, 0, 0], axis=-1)
    tile[black_pixels] = [255, 255, 255] 


    hsv_img = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2HSV)
    gray_mask = cv2.inRange(hsv_img, (0, 0, 70), (180, 10, 255))
    black_mask = cv2.inRange(hsv_img, (0, 0, 0), (180, 255, 85))
    # Set all grey/black pixels to white
    full_tile_bg = np.copy(tile)
    full_tile_bg[np.where(gray_mask | black_mask)] = 255

    # apply otsu mask first time for removing larger artifacts
    masked_image_gray = 255 * sk_color.rgb2gray(full_tile_bg)
    thresh = sk_filters.threshold_otsu(masked_image_gray)
    otsu_masking = masked_image_gray < thresh
    # improving mask
    #otsu_masking = sk_morphology.remove_small_objects(otsu_masking, 60)
    #otsu_masking = sk_morphology.dilation(otsu_masking, sk_morphology.square(12))
    #otsu_masking = sk_morphology.closing(otsu_masking, sk_morphology.square(5))
    #otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 250)
    tile = mask_rgb(tile, otsu_masking).astype(np.uint8)
    Image.fromarray(tile).save('tile.png')

    # apply otsu mask second time for removing small artifacts
    masked_image_gray = 255 * sk_color.rgb2gray(tile)
    thresh = sk_filters.threshold_otsu(masked_image_gray)
    otsu_masking = masked_image_gray < thresh
    otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 5000)
    otsu_thr = ~otsu_masking
    otsu_thr = otsu_thr.astype(np.uint8)

    #Image.fromarray(np.expand_dims(otsu_thr, axis=-1) * np.array([255, 255, 255]).astype(np.uint8)).save('otsu_thr.png')

    return otsu_thr


def create_splits(dest_dir, splits, K):
    # [[patien1], [patien2]]...

    #meta_df = meta_df[meta_df['id']]
    # [([], []), ] K (nb_split) x 2 x n
    os.makedirs(dest_dir, exist_ok=True)
    arr = [value for key, value in splits.items()]

    split_nb = 0
    for k in range(K):
        train_ids = arr.copy()
        del train_ids[k]
        train_ids = [arrss for arrs in train_ids for arrss in arrs]

        test_ids = np.array(arr[k]).flatten()
        #train_ids, test_ids = splits[k]

        data_train = np.column_stack((train_ids, [os.path.join('patches', id + '.h5') for id in train_ids], [os.path.join('adata', id + '.h5ad') for id in train_ids]))
        train_df = pd.DataFrame(data_train, columns=['sample_id', 'patches_path', 'expr_path'])

        data_test = np.column_stack((test_ids, [os.path.join('patches', id + '.h5') for id in test_ids], [os.path.join('adata', id + '.h5ad') for id in test_ids]))
        test_df = pd.DataFrame(data_test, columns=['sample_id', 'patches_path', 'expr_path'])
        train_df.to_csv(os.path.join(dest_dir, f'train_{k}.csv'), index=False)
        test_df.to_csv(os.path.join(dest_dir, f'test_{k}.csv'), index=False)


class WSI:
    wsi: openslide.OpenSlide = None
    contours_holes = None
    contours_tissue = None
    tissue_mask: np.ndarray = None


    def __init__(self, path):
        self.wsi = openslide.OpenSlide(path)


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
            width, height = self.wsi.dimensions
            downsample = target_width / width

            top_left = (0,0)
            #downsample = self.wsi.level_downsamples[vis_level]
            scale = [downsample, downsample]    
            #region_size = self.wsi.level_dimensions[vis_level]
            #img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
            img = np.array(self.wsi.get_thumbnail((width * downsample, height * downsample)))
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


    def dump_patches(
        self,
        patch_save_dir: str,
        adata: sc.AnnData, 
        src_pixel_size: float,
        name: str = None,
        patch_size_um: float=128,
        target_pixel_size: float=0.5,
        verbose=0,
        dump_visualization=True
    ):

        #TODO change
        #img = self.wsi.read_region((0, 0), 0, self.wsi.dimensions)
        #img = np.array(img)
        #if img.shape[2] == 4:
        #    img = img[:, :, :3]
        
        # minimum intersection percecentage with the tissue mask to keep a patch
        TISSUE_INTER_THRESH = 0.05
        TARGET_VIS_SIZE = 1000
        
        scale_factor = src_pixel_size / target_pixel_size
        patch_size_pxl = round(patch_size_um / src_pixel_size)
        patch_count = 0
        output_datafile = os.path.join(patch_save_dir, name + '.h5')

        assert len(adata.obs) == len(adata.obsm['spatial'])

        fig, ax = plt.subplots()
        
        mode_HE = 'w'
        i = 0
        img_width, img_height = self.wsi.dimensions
        patch_rectangles = [] # lower corner (x, y) + (widht, height)
        downscale_vis = TARGET_VIS_SIZE / img_width

        if self.tissue_mask is None:
            self._compute_mask()

        mask_plot = self.visualize_wsi(line_thickness=3, target_width=1000)

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
            
            #image_patch = img[yImage - patch_size_pxl // 2: yImage + patch_size_pxl // 2,
            #                    xImage - patch_size_pxl // 2: xImage + patch_size_pxl // 2, :]
            image_patch = self.wsi.read_region((xImage - patch_size_pxl // 2, yImage - patch_size_pxl // 2), 0, (patch_size_pxl, patch_size_pxl))
            rect_x = (xImage - patch_size_pxl // 2) * downscale_vis
            rect_y = (yImage - patch_size_pxl // 2) * downscale_vis
            rect_width = patch_size_pxl * downscale_vis
            rect_height = patch_size_pxl * downscale_vis

            image_patch = np.array(image_patch)
            if image_patch.shape[2] == 4:
                image_patch = image_patch[:, :, :3]


            patch_mask = self.tissue_mask[yImage - patch_size_pxl // 2: yImage + patch_size_pxl // 2,
                            xImage - patch_size_pxl // 2: xImage + patch_size_pxl // 2]
            patch_area = patch_mask.shape[0] ** 2
            pixel_count = patch_mask.sum()

            if pixel_count / patch_area < TISSUE_INTER_THRESH:
                continue

            patch_rectangles.append(matplotlib.patches.Rectangle((rect_x, rect_y), rect_width, rect_height))
            
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
        
        if dump_visualization:
            ax.add_collection(PatchCollection(patch_rectangles, facecolor='none', edgecolor='black', linewidth=0.3))
            ax.set_axis_off()
            os.makedirs(os.path.join(patch_save_dir, 'vis'), exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(patch_save_dir, 'vis', name + '.png'), dpi=400, bbox_inches = 'tight')
            
        if verbose:
            print(f'found {patch_count} valid patches')



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
    

    def _compute_mask(self):
        width, height = self.wsi.dimensions
        TARGET_WIDTH = 2000
        scale = TARGET_WIDTH / width
        thumbnail = np.array(self.wsi.get_thumbnail((width * scale, height * scale)))
        Image.fromarray(thumbnail).save('thumb.png')
        mask = apply_otsu_thresholding(thumbnail).astype(np.uint8)
        mask = 1 - mask
        self.tissue_mask = np.round(cv2.resize(mask, (height, width))).astype(np.uint8)
        mask_to_contours(self, self.tissue_mask)


    def compute_tissue_mask(self):

        wsi.dump_patches('/media/ssd2/hest/patches',
                            adata,
                            pixel_size,
                            id,
                            tissue_mask=mask,
                            verbose=1)


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
        #mask = np.load(mask_path)
        #mask = np.transpose(mask, (1, 0))
        wsi = WSI(img_path)


        wsi.dump_patches('/media/ssd2/hest/patches',
                           adata,
                           pixel_size,
                           id,
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
        
def mask_to_contours(wsi: WSI, mask: np.ndarray, keep_ids = [], exclude_ids=[], max_nb_holes=0):
    TARGET_EDGE_SIZE = 2000
    scale = TARGET_EDGE_SIZE / mask.shape[0]

    downscaled_mask = cv2.resize(mask, (round(mask.shape[0] * scale), round(mask.shape[1] * scale)))

    # Find and filter contours
    if max_nb_holes == 0:
        contours, hierarchy = cv2.findContours(downscaled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv2.findContours(downscaled_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
    #print('Num Contours Before Filtering:', len(contours))
    if hierarchy is None:
        hierarchy = []
    else:
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    filter_params = {
        'filter_color_mode': 'none',
        'max_n_holes': max_nb_holes,
        'a_t': 1000
    }

    seg_level = 0
    if filter_params: 
        foreground_contours, hole_contours = filter_contours(wsi.wsi, seg_level, contours, hierarchy, filter_params, scale)  # Necessary for filtering out artifacts

    
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
    
    
