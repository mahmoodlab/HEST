import pickle

import cv2
import numpy as np

try:
    import openslide
except ImportError:
    print("Couldn't import openslide, verify that openslide is installed on your system")
import scanpy as sc
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.measure as sk_measure
import skimage.morphology as sk_morphology
from tqdm import tqdm


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


def keep_largest_area(mask: np.ndarray) -> np.ndarray:
    label_image, num_labels = sk_measure.label(mask, background=0, return_num=True)
    largest_label = 0
    largest_area = 0
    for label in range(1, num_labels + 1):
        area = np.sum(label_image == label)
        if area > largest_area:
            largest_label = label
            largest_area = area
    largest_mask = np.zeros_like(mask, dtype=bool)
    largest_mask[label_image == largest_label] = True
    mask[~largest_mask] = 0
    return mask


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
    otsu_masking = sk_morphology.remove_small_objects(otsu_masking, 60)
    #otsu_masking = sk_morphology.dilation(otsu_masking, sk_morphology.square(12))
    #otsu_masking = sk_morphology.closing(otsu_masking, sk_morphology.square(5))
    #otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 250)
    tile = mask_rgb(tile, otsu_masking).astype(np.uint8)

    # apply otsu mask second time for removing small artifacts
    masked_image_gray = 255 * sk_color.rgb2gray(tile)
    thresh = sk_filters.threshold_otsu(masked_image_gray)
    otsu_masking = masked_image_gray < thresh
    otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 5000)
    otsu_thr = ~otsu_masking
    otsu_thr = otsu_thr.astype(np.uint8)

    #Image.fromarray(np.expand_dims(otsu_thr, axis=-1) * np.array([255, 255, 255]).astype(np.uint8)).save('otsu_thr.png')

    return otsu_thr


class WSI:
    wsi: openslide.OpenSlide = None
    contours_holes = None
    contours_tissue = None
    tissue_mask: np.ndarray = None


    def __init__(self, path):
        self.wsi = openslide.OpenSlide(path)
    


def filter_contours(contours, hierarchy, filter_params, scale):
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
                raise Exception()

    
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
        
        
def mask_to_contours(mask: np.ndarray, keep_ids = [], exclude_ids=[], max_nb_holes=0):
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

    if filter_params: 
        foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params, scale)  # Necessary for filtering out artifacts

    
    if len(foreground_contours) == 0:
        raise Exception('no contour detected')
    else:
        contours_tissue = scale_contour_dim(foreground_contours, 1 / scale)
        contours_holes = scale_holes_dim(hole_contours, 1 / scale)

    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

    contours_tissue = [contours_tissue[i] for i in contour_ids]
    contours_holes = [contours_holes[i] for i in contour_ids]

    #print('Num Contours After Filtering:', len(wsi.contours_tissue))


    return contours_tissue, contours_holes
    

def scale_holes_dim(contours, scale):
    r"""
    """
    return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]


def scale_contour_dim(contours, scale):
    r"""
    """
    return [np.array(cont * scale, dtype='int32') for cont in contours]
    
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
    
    
