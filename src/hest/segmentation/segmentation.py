import pickle
from functools import partial
from typing import Union

import cv2
import pandas as pd
from shapely import Polygon
import torch.nn.functional as F
import numpy as np
import torch
from geopandas import gpd
from huggingface_hub import snapshot_download
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from hest.segmentation.SegDataset import SegWSIDataset
from hest.utils import deprecated, get_path_relative
from hest.wsi import WSI, WSIPatcher, wsi_factory

try:
    import openslide
except Exception:
    print("Couldn't import openslide, verify that openslide is installed on your system, https://openslide.org/download/")
import scanpy as sc
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.measure as sk_measure
import skimage.morphology as sk_morphology

try:
    from cucim import CuImage
except ImportError:
    CuImage = None
    print("CuImage is not available. Ensure you have a GPU and cucim installed to use GPU acceleration.")

from tqdm import tqdm


def segment_tissue_deep(
    wsi: Union[np.ndarray, openslide.OpenSlide, 'CuImage', WSI],
    pixel_size: float,
    fast_mode=False,
    target_pxl_size=1,
    patch_size_um=512,
    model_name='deeplabv3_seg_v4.ckpt',
    batch_size=8,
    auto_download=True,
    num_workers=8
) -> gpd.GeoDataFrame:
    """ Segment the tissue using a DeepLabV3 model

    Args:
        wsi (Union[np.ndarray, openslide.OpenSlide, CuImage, WSI]): wsi
        pixel_size (float): pixel size in um/px for the wsi
        fast_mode (bool, optional): in fast mode the inference is done at 2 um/px instead of 1 um/px, 
            note that the inference pixel size is overwritten by the `target_pxl_size` argument if != 1. Defaults to False.
        target_pxl_size (int, optional): patches are scaled to this pixel size in um/px for inference. Defaults to 1.
        patch_size_um (int, optional): patch size in um. Defaults to 512.
        model_name (str, optional): model name in `HEST/models` dir. Defaults to 'deeplabv3_seg_v4.ckpt'.
        batch_size (int, optional): batch size for inference. Defaults to 8.
        auto_download (bool, optional): whenever to download the model weights automatically if not found. Defaults to True.
        num_workers (int, optional): number of workers for the dataloader during inference. Defaults to 8.

    Returns:
        gpd.GeoDataFrame: a geodataframe of the tissue contours, contains a column `tissue_id` indicating to which tissue the contour belongs to
    """
    pixel_size_src = pixel_size
    
    if fast_mode and target_pxl_size == 1:
        target_pxl_size = 2
    
    patch_size_deeplab = 512
    
    # TODO fix overlap
    overlap=0
    
    scale = pixel_size_src / target_pxl_size
    patch_size_src = round(patch_size_um / scale)
    wsi = wsi_factory(wsi)
    
    weights_path = get_path_relative(__file__, f'../../../models/{model_name}')
    
    patcher = WSIPatcher(wsi, patch_size_src, patch_size_deeplab)
        
    eval_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = SegWSIDataset(patcher, eval_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=2,
        kernel_size=1,
        stride=1
    )
    
    if auto_download:
        model_dir = get_path_relative(__file__, f'../../../models')
        snapshot_download(repo_id="MahmoodLab/hest-tissue-seg", repo_type='model', local_dir=model_dir, allow_patterns=model_name)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        
    new_state_dict = {}
    for key in checkpoint['state_dict']:
        if 'aux' in key:
            continue
        new_key = key.replace('model.', '')
        new_state_dict[new_key] = checkpoint['state_dict'][key]
    model.load_state_dict(new_state_dict)
    
    if torch.cuda.is_available():        
        model.cuda()
    
    model.eval()
    
    cols, rows = patcher.get_cols_rows()
    width, height = patch_size_deeplab * cols, patch_size_deeplab * rows
    stitched_img = np.zeros((height, width), dtype=np.uint8)
    src_to_deeplab_scale = patch_size_deeplab / patch_size_src
    
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            
            # coords are top left coords of patch
            imgs, coords = batch
            if torch.cuda.is_available(): 
                imgs = imgs.cuda()
            masks = model(imgs)['out']
            preds = masks.argmax(1).to(torch.uint8).detach()
            
            torch.cuda.synchronize()

            preds = preds.cpu().numpy()
            coords = np.column_stack((coords[0], coords[1]))
            
            # stitch the patches
            for i in range(preds.shape[0]):
                pred = preds[i]
                coord = coords[i]
                x, y = round(coord[0] * src_to_deeplab_scale), round(coord[1] * src_to_deeplab_scale)
                 
                y_end = min(y+patch_size_deeplab + overlap, height)
                x_end = min(x+patch_size_deeplab + overlap, width)
                stitched_img[y:y_end, x:x_end] += pred[:y_end-y, :x_end-x]
            
        
    mask = (stitched_img > 0).astype(np.uint8)
        
    gdf_contours = mask_to_contours(mask, max_nb_holes=5, pixel_size=pixel_size_src, contour_scale=1 / src_to_deeplab_scale)
        
    return gdf_contours


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


def contours_to_img(
    contours: gpd.GeoDataFrame, 
    img: np.ndarray, 
    draw_contours=False, 
    thickness=1, 
    downsample=1.,
    line_color=(0, 255, 0)
) -> np.ndarray:
    draw_cont = partial(cv2.drawContours, contourIdx=-1, thickness=thickness, lineType=cv2.LINE_8)
    draw_cont_fill = partial(cv2.drawContours, contourIdx=-1, thickness=cv2.FILLED)
    
    groups = contours.groupby('tissue_id')
    for _, group in groups:
        
        for _, row in group.iterrows():
            cont = np.array([[round(x * downsample), round(y * downsample)] for x, y in row.geometry.exterior.coords])
        
            if row['hole']:
                draw_cont_fill(image=img, contours=[cont], color=(0, 0, 0))
            else:
                draw_cont_fill(image=img, contours=[cont], color=line_color)
            if draw_contours:
                draw_cont(image=img, contours=[cont], color=line_color)
    return img


def get_tissue_vis(
            img: Union[np.ndarray, openslide.OpenSlide, 'CuImage', WSI],
            tissue_contours: gpd.GeoDataFrame,
            line_color=(0, 255, 0),
            line_thickness=5,
            target_width=1000,
            seg_display=True,
    ) -> Image:
        tissue_contours = tissue_contours.copy()
    
        wsi = wsi_factory(img)
    
        width, height = wsi.get_dimensions()
        downsample = target_width / width

        top_left = (0,0)
        
        img = wsi.get_thumbnail(round(width * downsample), round(height * downsample))

        if tissue_contours is None:
            return Image.fromarray(img)

        downscaled_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        downscaled_mask = np.expand_dims(downscaled_mask, axis=-1)
        downscaled_mask = downscaled_mask * np.array([0, 0, 0]).astype(np.uint8)

        if tissue_contours is not None and seg_display:
            downscaled_mask = contours_to_img(
                tissue_contours, 
                downscaled_mask, 
                draw_contours=True, 
                thickness=line_thickness, 
                downsample=downsample,
                line_color=line_color
            )

        alpha = 0.4
        img = cv2.addWeighted(img, 1 - alpha, downscaled_mask, alpha, 0)
        img = img.astype(np.uint8)

        return Image.fromarray(img)
    

@deprecated
def visualize_tissue_seg(
            img,
            tissue_mask,
            contours_tissue,
            contour_holes,
            line_color=(0, 255, 0),
            hole_color=(0, 0, 255),
            line_thickness=5,
            target_width=1000,
            seg_display=True,
    ):
        hole_fill_color = (0, 0, 0)
    
    
        wsi = wsi_factory(img)
    
        width, height = wsi.get_dimensions()
        downsample = target_width / width

        top_left = (0,0)
        scale = [downsample, downsample]    
        
        img = wsi.get_thumbnail(round(width * downsample), round(height * downsample))
        #img = cv2.resize(img, (round(width * downsample), round(height * downsample)))
        if tissue_mask is None and contours_tissue is None and contour_holes is None:
            return Image.fromarray(img)

        downscaled_mask = cv2.resize(tissue_mask, (img.shape[1], img.shape[0]))
        downscaled_mask = np.expand_dims(downscaled_mask, axis=-1)
        downscaled_mask = downscaled_mask * np.array([0, 0, 0]).astype(np.uint8)


        draw_cont = partial(cv2.drawContours, contourIdx=-1, thickness=line_thickness, lineType=cv2.LINE_8)
        draw_cont_fill = partial(cv2.drawContours, contourIdx=-1, thickness=cv2.FILLED)

        if contours_tissue is not None and seg_display:
            for _, cont in enumerate(contours_tissue):
                cont = np.array(scale_contour_dim(cont, scale))
                draw_cont(image=img, contours=[cont], color=line_color)
                draw_cont_fill(image=downscaled_mask, contours=[cont], color=line_color)

            ### Draw hole contours
            for cont in contour_holes:
                cont = scale_contour_dim(cont, scale)
                draw_cont(image=img, contours=cont, color=hole_color)
                draw_cont_fill(image=downscaled_mask, contours=cont, color=hole_fill_color)

        alpha = 0.4
        downscaled_mask = downscaled_mask
        tissue_mask = cv2.resize(downscaled_mask, (width, height)).round().astype(np.uint8)
        img = cv2.addWeighted(img, 1 - alpha, downscaled_mask, alpha, 0)
        img = img.astype(np.uint8)

        return Image.fromarray(img)


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


def filter_contours(contours, hierarchy, filter_params, scale, pixel_size):
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
        a *= pixel_size ** 2

        if a == 0: continue

        
        
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            
            if (filter_params['filter_color_mode'] == 'none') or (filter_params['filter_color_mode'] is None):
                filtered.append(cont_idx)
                holes = [hole_idx for hole_idx in holes if cv2.contourArea(contours[hole_idx]) * pixel_size ** 2 > filter_params['min_hole_area']]
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
        filtered_holes = unfilered_holes[:filter_params['max_n_holes']]
        #filtered_holes = []
        
        # filter these holes
        #for hole in unfilered_holes:
        #    if cv2.contourArea(hole) > filter_params['a_h']:
        #        filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours
        
        
def mask_to_contours(mask: np.ndarray, keep_ids = [], exclude_ids=[], max_nb_holes=0, min_contour_area=1000, pixel_size=1, contour_scale=1.):
    TARGET_EDGE_SIZE = 2000
    scale = TARGET_EDGE_SIZE / mask.shape[0]

    downscaled_mask = cv2.resize(mask, (round(mask.shape[1] * scale), round(mask.shape[0] * scale)))

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
        'a_t': min_contour_area * pixel_size ** 2,
        'min_hole_area': 4000 * pixel_size ** 2
    }

    if filter_params: 
        foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params, scale, pixel_size)  # Necessary for filtering out artifacts

    
    if len(foreground_contours) == 0:
        raise Exception('no contour detected')
    else:
        contours_tissue = scale_contour_dim(foreground_contours, contour_scale / scale)
        contours_holes = scale_holes_dim(hole_contours, contour_scale / scale)

    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

    tissue_poly = [Polygon(contours_tissue[i].squeeze(1)) for i in contour_ids]
    hole_poly = [Polygon(contours_holes[i][0].squeeze(1)) for i in contour_ids if len(contours_holes[i]) > 0]
    geometry = tissue_poly + hole_poly
    tissue_ids = [i for i in contour_ids] + [i for i in contour_ids if len(contours_holes[i]) > 0]
    tissue_types = ['tissue' for _ in contour_ids] + ['hole' for i in contour_ids if len(contours_holes[i]) > 0]
    
    gdf_contours = gpd.GeoDataFrame(pd.DataFrame(tissue_ids, columns=['tissue_id']), geometry=geometry)
    gdf_contours['hole'] = tissue_types
    gdf_contours['hole'] = gdf_contours['hole'] == 'hole'
    
    return gdf_contours
    

def scale_holes_dim(contours, scale):
    r"""
    """
    return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]


def scale_contour_dim(contours, scale):
    r"""
    """
    return [np.array(cont * scale, dtype='int32') for cont in contours]