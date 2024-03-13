import os

import pandas as pd
import numpy as np
import scanpy as sc
from PIL import Image
from PIL.TiffTags import TAGS
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
from scipy import ndimage as ndi
import cv2
from skimage import transform
import rasterio
from rasterio import warp
from kwimage.im_cv2 import warp_affine, imresize
from kwimage.transform import Affine
import json
import urllib.request
import tarfile
import requests
import subprocess
from threading import Thread
from time import sleep


def _extract_tar_gz(file_path, extract_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def _download_from_row(row, root_path):
    col_to_download = [
        'image_link', 'xenium_bundle_link', 'spatial_data_link', 
        'filtered_count_h5_link', 'alignment_file_link'
    ]
    for col in col_to_download:
        if row[col] is not None and not isinstance(row[col], float):
            dest_path = os.path.join(root_path, row[col].split('/')[-1])
            if col != 'image_link':
                continue
            if dest_path.endswith('.tar.gz'):
                subprocess.run(['wget', row[col], '-O', dest_path], check=True)
            else:
                subprocess.Popen(['wget', row[col], '-O', dest_path])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #r = requests.get(row[col]) 
            #with open(dest_path, 'wb') as outfile:
            #    outfile.write(r.content)
            #urllib.request.urlretrieve(row[col], dest_path)

            if dest_path.endswith('.tar.gz'):
                _extract_tar_gz(dest_path, os.path.dirname(dest_path))
                

def download_from_meta_df(meta_df: pd.DataFrame, samples_folder: str):
    threads = []
    for index, row in meta_df.iterrows():
        dirname = row['dataset_title']
        if row['image'] == False:
            continue
        tech = 'visium' if 'visium' in row['10x Instrument(s)'].lower() else 'xenium'
        dir_path = os.path.join(samples_folder, tech, dirname)
        root_path = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if row['subseries'] is not None and not np.isnan(row['subseries']):
            subseries_path = os.path.join(dir_path, row['subseries'])
            root_path = subseries_path
            if not os.path.exists(subseries_path):
                os.makedirs(subseries_path)
                
        col_to_download = [
            'image_link', 'xenium_bundle_link', 'spatial_data_link', 
            'filtered_count_h5_link', 'alignment_file_link'
        ]

        _download_from_row(row, root_path)

        



def filter_st_data(adata, nb_hvgs, min_counts = 5000, max_counts = 35000, 
                            pct_counts_mt = 20, min_cells = 10
                            ):
    """Filters Spatial Transcriptomics data
    
    This method filters previously loaded Spatial Transcriptomics adata by applying
    the following transformations:
    - filtering out spots that contain less than `min_counts` or more than `max_counts` transcripts
    - filtering out spots with a count of mitochondrial transcripts being more or equal to `pct_counts_mt`
    - filtering out genes that are expressed in strictly less than `min_cells` spots
    - normalizing the spots by count per cell
    - take the log1p of the expressions
    - keep only the `nb_hvgs` most highly variable genes
    """
    
    # QC: Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
    adata.var_names_make_unique()
    # High mitochondrial genes indicates low quality cells
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # QC: Filter out cells and genes
    n_c_before = adata.n_obs
    print(f"# cells before filtering: {n_c_before}")
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    adata = adata[adata.obs["pct_counts_mt"] < pct_counts_mt]
    n_c_after = adata.n_obs
    print(f"# cells after filter: {n_c_after}")
    print(f"# cells removed: {n_c_before - n_c_after}")
    gene_list = adata.var_names
    sc.pp.filter_genes(adata, min_cells=min_cells) # only keep genes expressed in more than 10 cells
    gene_filtered = list(set(gene_list) - set(adata.var_names))
    print(f"# genes removed: {len(gene_filtered)}")

    # Log Normalization
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    # select highly variable genes (hvgs)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=nb_hvgs)

    
    return adata, gene_filtered


def downsample_image(img_path, target_resolution = 10, save_image = False, output_path = None):
    """
    downsample the image from 20x to 10x or 2.5x
    10x Visium image is 20x by default,
    here we provide flexbility of downsampling to 10x or 2.5x
    """
    # Check target resolution is valid
    if target_resolution not in [10, 2.5]:
        raise ValueError("Target resolution should be 10 (10x) or 2.5 (2.5x).")

    # Open the 20x image
    Image.MAX_IMAGE_PIXELS = None # need this config to open large image
    img_20x = Image.open(img_path)
    print(f"Original image size: {img_20x.width} x {img_20x.height}")

    print(f"Downsampling to {target_resolution}x...")

    # Calculate the downsampling factor
    downsample_factor = 20 / target_resolution

    # Calculate the new size
    new_size = (int(img_20x.width // downsample_factor), int(img_20x.height // downsample_factor))

    # Downsample the image
    img_downsampled = img_20x.resize(new_size, Image.ANTIALIAS)
    
    # if save_image:
    # # Save the downsampled image
    #     img_downsampled.save(f'{target_resolution}_x_downsampled.tif')
    
    return img_downsampled


def read_any(path):
    if 'visium' in path.lower():
        img_filename = None
        for file in os.listdir(path):
            if file.endswith('.tif') or file.endswith('.btf'):
                img_filename = file
                break
            
        feature_file = None
        for file in os.listdir(path):
            if file.endswith('filtered_feature_bc_matrix.h5'):
                feature_file = file
                break
            
        alignment_file = None
        for file in os.listdir(path):
            if file.endswith('alignment_file.json'):
                alignment_file = file
                break
            
        adata, _, img = read_10x_visium(
            st_gene_expression_path=os.path.join(path, feature_file),
            spatial_coord_path=os.path.join(path, 'spatial/tissue_positions.csv'),
            img_path=os.path.join(path, img_filename),
            alignment_file_path=os.path.join(path, alignment_file)
        )
        
        return adata, img
        
    elif 'xenium'in path.lower():
        for file in os.listdir(path):
            if file.endswith('.tif'):
                if file not in ['morphology.ome.tif', 'morphology_focus.ome.tif', 'morphology_mip.ome.tif']:
                    img_filename = file
                    break
                
        alignment_path = None
        for file in os.listdir(path):
            if file.endswith('he_imagealignment.csv'):
                alignment_path = os.path.join(path, file)
        
        adata, img = read_10x_xenium(
            feature_matrix_path=os.path.join(path, 'cell_feature_matrix.h5'), 
            transcripts_path=os.path.join(path, 'transcripts.parquet'), 
            img_path=os.path.join(path, img_filename), 
            alignment_file_path=alignment_path, 
            in_tissue_only = True
        )
        
        return adata, img
      

def _register_downscale_img(adata, img):
    print('image size is ', img.shape)
    downscale_factor = 0.025
    downscaled_fullres = imresize(img, downscale_factor)
    
    
    # register the image
    adata.uns['spatial'] = {}
    adata.uns['spatial']['V1_Adult_Mouse_Brain'] = {}
    adata.uns['spatial']['V1_Adult_Mouse_Brain']['images'] = {}
    adata.uns['spatial']['V1_Adult_Mouse_Brain']['images']['downscaled_fullres'] = downscaled_fullres
    adata.uns['spatial']['V1_Adult_Mouse_Brain']['scalefactors'] = {}
    adata.uns['spatial']['V1_Adult_Mouse_Brain']['scalefactors']['spot_diameter_fullres'] = 55.
    adata.uns['spatial']['V1_Adult_Mouse_Brain']['scalefactors']['tissue_downscaled_fullres_scalef'] = downscale_factor
  

def read_10x_visium(
    st_gene_expression_path: str, 
    spatial_coord_path: str, 
    img_path: str, 
    alignment_file_path: str = None, 
    in_tissue_only = True, 
    downsample_factor = None
):
    """
    Read the spatial data from 10x Visium platform and filter out spots that are not under tissue
    st_gene_expression_path: path to the gene expression matrix (.h5 file)
    spatial_coord_path: path to the spatial coordinates (e.g dataV1_Human_Lymph_Node/spatial/tissue_positions_list.csv)
    img_path: path to the H&E image
    """
    
    ### 1. GENE EXPRESSION
    adata = sc.read_10x_h5(st_gene_expression_path)
    adata.var_names_make_unique()
    print(adata)

    ### 2. SPATIAL COORDINATES
    spatial = pd.read_csv(spatial_coord_path, sep=",", na_filter=False, index_col=0) 
    #spatial = spatial.rename(columns={1: "in_tissue", # in_tissue: 1 if spot is captured in tissue region, 0 otherwise
    #                                2: "x_array", # spot row index
    #                                3: "y_array", # spot column index
    #                                4: "x_pixel", # spot x coordinate in image pixel
    #                                5: "y_pixel"}) # spot y coordinate in image pixel
    
    # make sure the spot barcodes aligned
    # sometimes spatial can be a greater set, but mostly they are the same
    barcode_diff = len(spatial) - len(adata.obs) 
    if barcode_diff != 0:
        print(f'{barcode_diff} spots are not aligned, adata might have been filtered within tissue region already')
    else:
        print(f'All {len(adata.obs)} spots are aligned with spatial coordinates')
    # align + match order        
    spatial_aligned = spatial.reindex(adata.obs.index)
    assert np.array_equal(spatial_aligned.index, adata.obs.index)
    
    print(spatial_aligned)
    
    col1 = spatial_aligned['pxl_col_in_fullres'].values
    col2 = spatial_aligned['pxl_row_in_fullres'].values
    
    matrix = np.vstack((col1, col2)).T
    
    adata.obsm['spatial'] = matrix
    

    ### More adata processing
    # add into adata.obs
    adata.obs = spatial_aligned
    # filter out spots outside tissue region
    print(f'Before filtering, {len(adata.obs)} spots are in adata')
    if in_tissue_only:
        adata = adata[adata.obs["in_tissue"] == 1]
        
    ### 3. H&E IMAGE
    if downsample_factor:
        img = np.array(downsample_image(img_path, downsample_factor))
    else:
        img = np.array(Image.open(img_path))

        
    if alignment_file_path is not None:
        f = open(alignment_file_path)
        
        data = json.load(f)
        #alignment_matrix = np.array(data['transform'])
        df = pd.DataFrame(data['oligo'])
        df = df.rename(columns={
            'row': 'array_row',
            'col': 'array_col',
            'imageX': 'pxl_col_in_fullres',
            'imageY': 'pxl_row_in_fullres'
        })
        df_merged = spatial_aligned.rename(columns={
            'pxl_col_in_fullres': 'pxl_col_in_fullres_old',
            'pxl_row_in_fullres': 'pxl_row_in_fullres_old'
        })
        
        matrix_cyt = np.array(data['cytAssistInfo']['transformImages'])
        
        df = df[['array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]
        #original_index = df_merged.index
        df_merged = df_merged.merge(df, on=['array_row', 'array_col'], how='left')
        
        adata.obs = df_merged
        
        col1 = df_merged['pxl_col_in_fullres'].values
        col2 = df_merged['pxl_row_in_fullres'].values
        
        #matrix = (np.vstack((col1, col2, np.ones(len(col1)))).T# @ matrix_cyt.T)[:,:2]
        #matrix = (np.vstack((col1, col2))).T
        cytassist_to_fullres = np.linalg.inv(matrix_cyt)
        
        matrix = (np.vstack((col1, col2, np.ones(len(col1)))).T @ cytassist_to_fullres.T)[:,:2]
        
        adata.obsm['spatial'] = matrix 
        
        
        #affine = Affine(matrix_cyt)
        
        #img2 = warp_affine(img, affine, dsize='auto', large_warp_dim=1000)
        #img = img2
        
        #plt.imshow(img)
        #plt.show()
        
        #f.close()
        
    # register the image
    _register_downscale_img(adata, img)

    return adata, spatial_aligned, img


def xenium_to_pseudo_visium(adata, df: pd.DataFrame, pixel_size):
    y_max = df['y_location'].max()
    y_min = df['y_location'].min()
    x_max = df['x_location'].max()
    x_min = df['x_location'].min()
    
    m = math.ceil((y_max - y_min) / (100 / pixel_size))
    n = math.ceil((x_max - x_min) / (100 / pixel_size))
    
    features = df['feature_name'].unique()
    
    spot_grid = pd.DataFrame(0, index=range(m * n), columns=features)
    #spot_grid = pd.DataFrame(0, index=range(m * n), columns=features)
    
    a = np.floor((df['x_location'] - x_min) / (100. / pixel_size)).astype(int)
    b = np.floor((df['y_location'] - y_min) / (100. / pixel_size)).astype(int)
    
    c = b * n + a
    features = df['feature_name']
    
    cols = spot_grid.columns.get_indexer(features)
    
    spot_grid_np = spot_grid.values.astype(np.uint16)
    spot_grid_np[c, cols] += 1
    
    df = pd.DataFrame(spot_grid_np, columns=spot_grid.columns)
    df['x'] = x_min + (df.index % n) * (100. / pixel_size) + (50. / pixel_size)
    df['y'] = y_min + np.floor(df.index / n) * (100. / pixel_size) + (50. / pixel_size)
    return df


def read_10x_xenium(
    feature_matrix_path: str, 
    #cell_csv_path: str, 
    transcripts_path: str,
    img_path: str, 
    alignment_file_path: str, 
    in_tissue_only = True
):
    CACHED_FILENAME = 'cached_aligned_image.tif'
    
    adata = sc.read_10x_h5(
        filename=feature_matrix_path
    )
    
    cur_dir = os.path.dirname(transcripts_path)    
    
    experiment_file = open(os.path.join(cur_dir, 'experiment.xenium'))
    dict = json.load(experiment_file)

    pixel_size = dict['pixel_size']
    
    
    df_transcripts = pd.read_parquet(transcripts_path)
    
    #df = pd.read_csv(
    #    cell_csv_path
    #)
    
    
    #df.set_index(adata.obs_names, inplace=True)
    #adata.obs = df.copy()
    
    #adata.obsm['transcripts'] = df_transcripts
    #adata.obs = df_transcripts
    
    #adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()
    
    df_transcripts["x_location"] = df_transcripts["x_location"] / pixel_size
    df_transcripts["y_location"] = df_transcripts["y_location"] / pixel_size
    
    spots_df = xenium_to_pseudo_visium(adata, df_transcripts, pixel_size)
    
    
    # convert from micrometers to pixels and register to Anndata object
   # adata.obsm["spatial"] = df_transcripts[["x_location", "y_location"]].copy().to_numpy() / pixel_size
    
    print(adata.obs)
    
    #cprobes = (
    #    adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100
    #)
    #cwords = (
    #    adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100
    #)
    #print(f"Negative DNA probe count % : {cprobes}")
    #print(f"Negative decoding count % : {cwords}")
    
    
    
    #print('positions are ', adata.obsm['spatial'])
    
    if alignment_file_path is not None:
        # check for a cached transformed file (as affine transformations of WSIs are very expensive)
        cached_image_path = None
        for file in os.listdir(cur_dir):
            if CACHED_FILENAME in file:
                cached_image_path = os.path.join(cur_dir, file)
        
        if cached_image_path is not None:
            img = tifffile.imread(cached_image_path)
        else:
            img = tifffile.imread(img_path)
            
            alignment_matrix = pd.read_csv(alignment_file_path, header=None).values
            print('detected alignment file...')
            print(alignment_matrix)
        
            #resized_img = imresize(img, 0.2125)
            #alignment_matrix[0][2] *= 0.2125
            #alignment_matrix[1][2] *= 0.2125

            affine = Affine(alignment_matrix)
        
            # Probably need to setup some extra memory swap before performing this step
            # will probably run out of memory otherwise
            img2 = warp_affine(img, affine, dsize='auto', large_warp_dim='auto')
            img = img2
            cached_path = os.path.join(cur_dir, CACHED_FILENAME)
            tifffile.imwrite(cached_path, img, photometric='rgb')
    else:
        img = tifffile.imread(img_path)
            
            
    _register_downscale_img(adata, img)

    #if downsample_factor:
    #    img = np.array(downsample_image(img_path, downsample_factor))
    #else:
    #    img = np.array(Image.open(img_path))
    
    downscale_factor = 0.025
    img2 = imresize(img, downscale_factor)
    
    plt.imshow(img2)
    my_df = spots_df[spots_df['GPC3'] != 0]
    plt.scatter(my_df['x'] * downscale_factor, my_df['y'] * downscale_factor, s=10)
    plt.show()
        
    return adata, img


def extract_patch_expression_pairs(adata, image, patch_size = 200):
    # df is a pandas DataFrame containing pxl_row_in_fullres and y_pixel columns
    # image is the histology image (e.g., in RGB format)
    # patch_size is the pixel size of the square patch to extract

    patches = []
    patch_half = patch_size // 2

    max_x = image.shape[0]
    max_y = image.shape[1]

    # make sure our image is correct
    assert df['pxl_row_in_fullres'].max() <= max_x and df['pxl_col_in_fullres'].max() <= max_y

    # denormalize the expression and spot position dataframes for fast lookup
    exp_df = adata.to_df()
    exp_df['barcode'] = exp_df.index
    join_df = exp_df.join(df, on='barcode', how='left')
    mask = list(exp_df.columns)
    mask.append('pxl_row_in_fullres')
    mask.append('pxl_col_in_fullres')
    join_df = join_df[mask]
    
    for barcode, row in tqdm(join_df.iterrows()):
        x_pixel = row['pxl_row_in_fullres']
        y_pixel = row['pxl_col_in_fullres']
        
        patch = image[max(0, x_pixel - patch_half):min(max_x, x_pixel + patch_half + 1),
                      max(0, y_pixel - patch_half):min(max_y, y_pixel + patch_half + 1)]

        patches.append(patch)


    join_df['image'] = patches

    return join_df


def extract_image_patches(adata: sc.AnnData, image, patch_size = 200):
    # df is a pandas DataFrame containing x_pixel and y_pixel columns
    # image is the histology image (e.g., in RGB format)
    # patch_size is the pixel size of the square patch to extract
    
    spots_coordinates = adata.obsm['spatial'] # coordinates of the spots in full res (px)

    patches = []
    patch_half = patch_size // 2

    max_x = image.shape[0]
    max_y = image.shape[1]

    # make sure our image is correct
    assert int(spots_coordinates[:, 1].max()) <= max_x and int(spots_coordinates[:, 0].max()) <= max_y

    for row in spots_coordinates:
        x_pixel = int(row[1])
        y_pixel = int(row[0])
        
        patch = image[max(0, x_pixel - patch_half):min(max_x, x_pixel + patch_half + 1),
                      max(0, y_pixel - patch_half):min(max_y, y_pixel + patch_half + 1)]

        patches.append(patch)

    return patches


def visualize_patches(patches, cols=5, figsize=(15, 15)):
    rows = math.ceil(len(patches) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    c_empty = 0
    for i, patch in enumerate(tqdm(patches)):
        if patch.size > 0:  # Check if the patch is not empty
            axes[i].imshow(patch)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        else:
            axes[i].set_visible(False)  # Hide the axis if the patch is empty
            c_empty += 1
    print(f'Number of empty patches: {c_empty}')
    
    plt.tight_layout()
    plt.show()