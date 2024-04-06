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
from kwimage.im_cv2 import warp_affine, imresize
from kwimage.transform import Affine
import json
import urllib.request
import tarfile
import subprocess
from threading import Thread
from time import sleep
import h5py
import shutil
import gzip
from scipy import sparse
from .align import autoalign_with_fiducials
import seaborn as sns



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
            if col != 'filtered_count_h5_link':
                continue
            if dest_path.endswith('.tar.gz'):
                subprocess.run(['wget', row[col], '-O', dest_path], check=True)
            else:
                subprocess.run(['wget', row[col], '-O', dest_path], check=True)
                #subprocess.Popen(['wget', row[col], '-O', dest_path])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def _find_first_file_endswith(dir, suffix, exclude=''):
    files_dir = os.listdir(dir)
    matching = [file for file in files_dir if file.endswith(suffix) and file != exclude]
    if len(matching) == 0:
        return None
    else:
        return os.path.join(dir, matching[0])
   
   
def _get_path_from_meta_row(row, root_path):
    tech = 'visium' if 'visium' in row['10x Instrument(s)'].lower() else 'xenium'
    if pd.isnull(row['subseries']):
        return os.path.join(root_path, tech, row['dataset_title'])
    else:
        return os.path.join(root_path, tech, row['dataset_title'], row['subseries'])
    

def _get_name_from_meta_row(row):
    tech = 'visium' if 'visium' in row['10x Instrument(s)'].lower() else 'xenium'
    if pd.isnull(row['subseries']):
        return row['dataset_title'] + '_' + tech
    else:
        return row['dataset_title'] + '_' + row['subseries'] + '_' + tech
 

def save_metrics_plot(adata, save_path, name):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(adata.obs['log1p_n_genes_by_counts'], bins=50, kde=False, color='blue', ax=ax)
    plt.savefig(save_path)
    plt.close()

def save_spatial_metrics_plot(adata, save_path, name, filtered_adata):
    print('Plotting metrics pol...')
    
    fig, ax = plt.subplots(figsize=(30, 15))
    ax = [[None for _ in range(6)] for _ in range(4)]
    
    ax[0][0] = plt.subplot2grid((6, 6), (3, 0), colspan=3)
    ax[0][1] = plt.subplot2grid((6, 6), (4, 0), colspan=3)
    ax[0][2] = plt.subplot2grid((6, 6), (5, 0), colspan=3)
    ax[1][0] = plt.subplot2grid((6, 6), (3, 3), colspan=3)
    ax[1][1] = plt.subplot2grid((6, 6), (4, 3), colspan=3)
    ax[1][2] = plt.subplot2grid((6, 6), (5, 3), colspan=3)
    ax[1][3] = plt.subplot2grid((6, 6), (0, 0), rowspan=3, colspan=3)
    ax[1][4] = plt.subplot2grid((6, 6), (0, 3), rowspan=3, colspan=3)
    
    my_filtered_adata = adata.copy()
    #missing_obs_idx = pd.concat([adata.obs, filtered_adata.obs]).drop_duplicates(keep=False).index
    my_filtered_adata.obs['filtered_out'] = [0 for _ in range(len(my_filtered_adata.obs))]
    #my_filtered_adata.obs[missing_obs_idx] = 1
    #adata.obs['filtered_out'] = [0 for _ in range(len(adata.obs))]
    
    sns.histplot(adata.obs['log1p_n_genes_by_counts'], bins=50, kde=False, color='blue', ax=ax[0][0])

    
    sns.histplot(adata.obs['pct_counts_mito'], bins=50, kde=False, color='blue', ax=ax[0][1])
    
    sns.histplot(adata.obs['log1p_total_counts'], bins=50, kde=False, color='blue', ax=ax[0][2])
    #sc.pl.spatial(adata, show=None, img_key="downscaled_fullres", color=['total_counts'], title=f"total_counts", ax=ax[0][3])
    
    #sc.pl.spatial(adata, show=None, img_key="downscaled_fullres", color=['filtered_out'], title=f"filtered_out", ax=ax[0][4], color_map='rainbow')

    sns.histplot(filtered_adata.obs['log1p_n_genes_by_counts'], bins=50, kde=False, color='blue', ax=ax[1][0])
    

    sns.histplot(filtered_adata.obs['pct_counts_mito'], bins=50, kde=False, color='blue', ax=ax[1][1])
    sns.histplot(filtered_adata.obs['log1p_total_counts'], bins=50, kde=False, color='blue', ax=ax[1][2])
    
    sc.pl.spatial(adata, show=None, img_key="downscaled_fullres", color=['total_counts'], title=f"total_counts before filtering", ax=ax[1][3])
    sc.pl.spatial(my_filtered_adata, show=None, img_key="downscaled_fullres", color=['filtered_out'], title=f"filtered out spots", ax=ax[1][4], color_map='rainbow')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'metrics_plot.png'))
    plt.close()
    
    
def filter_adata(adata):
    filtered_adata = adata.copy()
    std = filtered_adata.obs['total_counts'].std()
    mean = filtered_adata.obs['total_counts'].mean()
    lower_cutoff = mean - std  # Adjust this value as needed
    higher_cutoff = mean + std  # Adjust this value as needed

    # Filter cells based on total count
    sc.pp.filter_cells(adata, min_counts=lower_cutoff)
    sc.pp.filter_cells(adata, max_counts=higher_cutoff)
    
    
    return filtered_adata
    

def save_spatial_plot(adata, save_path, name, processed=False):
    print("Plotting spatial plots...")


    #print(sample_name)
    
    #plt.imshow(adata.uns['spatial']['ST']['images']['downscaled_fullres'])
    #plt.show()
    
    if processed:
        sc.pl.spatial(adata, img_key="downscaled_fullres", color=["total_counts", "n_genes_by_counts", "pct_counts_in_top_200_genes"],
                    ncols=3, cmap='plasma', alpha_img=0.5, title=[f"{name} total_counts", "n_genes_by_counts", "pct_counts_in_top_200_genes"], show=False)
    else:       
        sc.pl.spatial(adata, show=None, img_key="downscaled_fullres", color=['total_counts'], title=f"{name} in_tissue spots")
    
    # Generate spatial plots without showing them
    
    #fig = plt.figure()

    # Adjust the layout to make room for the custom titles
    #fig.tight_layout()
    #fig.add_axes(ax)
    
    #plt.show()
    
    if processed:
        filename = f"processed_spatial_plots.png"
    else:
        filename = f"spatial_plots.png"
    
    # Save the figure
    plt.savefig(os.path.join(save_path, filename))
    plt.close()  # Close the plot to free memory
    print(f"H&E overlay spatial plots saved in {save_path}")


def write_wsi(img, save_path, meta_dict):
    pixel_size = meta_dict['pixel_size_um_estimated']
    pixel_size_embedded = meta_dict['pixel_size_um_embedded']
    
    with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
        extratags = {
            'EstimatedPhysicalSizeX': f"{pixel_size} µm",
            'EstimatedPhysicalSizeY': f"{pixel_size} µm",
            'EmbeddedPhysicalSizeX': f"{pixel_size_embedded} µm",
            'EmbeddedPhysicalSizeY': f"{pixel_size_embedded} µm",            
        }
        extratags = json.dumps(extratags)
        
        metadata = {
         'PhysicalSizeX': pixel_size,
         'PhysicalSizeXUnit': 'µm',
         'PhysicalSizeY': pixel_size,
         'PhysicalSizeYUnit': 'µm'
        }
        options = dict(
            tile=(256, 256), 
            compression='deflate', 
            metadata=metadata,
            resolution=(
                1. / pixel_size,
                1. / pixel_size
            )
        )
        tif.write(img, subifds=3, description=extratags, **options)

        # save pyramid levels to the two subifds
        # in production use resampling to generate sub-resolutions
        tif.write(img[::2, ::2], subfiletype=1, **options)
        tif.write(img[::4, ::4], subfiletype=1, **options)
        tif.write(img[::8, ::8], subfiletype=1, **options)


    
def process_all(meta_df, root_path, save_plots=True):
    #path_list = [] 
    #for root, d_names, f_names in os.walk(root_dir):
    #    if len(f_names) == 0 or os.path.basename(root) == 'spatial':
    #        continue
    #    path_list.append(root)
    
    adata_list = []
    img_list = []
    raw_bc_matrices = []
    sample_names = []
    tissue_positions_df_list = []
    for _, row in tqdm(meta_df.iterrows()):
        if not row['image']:
            continue
        path = _get_path_from_meta_row(row, root_path)
        name = _get_name_from_meta_row(row)
        sample_names.append(name)
        adata, tissue_positions_df, img, raw_bc_matrix = read_any(path)
        #processed_adata, _ = filter_st_data(adata, 500, min_counts=5000, max_counts=35000, 
        #    pct_counts_mt=20, min_cells=10)
        adata_list.append(adata)
        tissue_positions_df_list.append(tissue_positions_df)
        img_list.append(img)
        raw_bc_matrices.append(raw_bc_matrix)
    
        #save_aligned_data(path, adata, tissue_positions_df, img, raw_bc_matrix)
        if save_plots:
            save_spatial_plot(adata, os.path.join(path, 'processed'), name)
            #save_spatial_plot(processed_adata, os.path.join(path, 'aligned'), name, processed=True)
        
    
    return adata_list, img_list, sample_names


def _find_biggest_img(path):
    ACCEPTED_FORMATS = ['.tif', '.jpg', '.btf', '.png', '.tiff', '.TIF']
    biggest_size = -1
    biggest_img_filename = None
    for file in os.listdir(path):
        ls = [s for s in ACCEPTED_FORMATS if file.endswith(s)]
        if len(ls) > 0:
            if file not in ['aligned_fullres_HE.ome.tif', 'morphology.ome.tif', 'morphology_focus.ome.tif', 'morphology_mip.ome.tif']:
                size = os.path.getsize(os.path.join(path, file))
                if size > biggest_size:
                    biggest_img_filename = file
                    biggest_size = size
    if biggest_img_filename is None:
        raise Exception(f"Couldn't find an image automatically, make sure that the folder {path} contains an image of one of these types: {ACCEPTED_FORMATS}")
    return biggest_img_filename


def join_object_to_adatas_GSE214989(path):
    adata = sc.read_10x_h5(path)
    sampleIDS = ['_1', '_2', '_3']
    for sampleID in sampleIDS:
        my_adata = adata.copy()
        df = my_adata.obs
        df = df[df.index.str.endswith(sampleID)]
        new_df = my_adata.to_df().loc[df.index]
        new_df.index = [idx[:-2] for idx in new_df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'{sampleID}_filtered_feature_bc_matrix.h5'))


def join_object_to_adatas_GSE171351(path):
    adata = sc.read_h5ad(path)
    sampleIDS = ['A1', 'B1', 'C1', 'D1']
    for sampleID in sampleIDS:
        my_adata = adata.copy()
        df = my_adata.obs#.reset_index(drop=True)
        df = df[df['sampleID'] == sampleID]
        new_df = my_adata.to_df().loc[df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        new_adata.obs = my_adata.obs[my_adata.obs['sampleID'] == sampleID]
        
        
        new_adata.uns['spatial'] = my_adata.uns['spatial'][sampleID]
        
        #col1 = new_adata.obs['pxl_col_in_fullres'].values
        #col2 = new_adata.obs['pxl_row_in_fullres'].values
        #matrix = (np.vstack((col1, col2))).T
        
        #new_adata.obsm['spatial'] = matrix 
        
        #adatas.append(new_adata)
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'{sampleID}_filtered_feature_bc_matrix.h5'))


def _ST_spot_to_pixel(x, y, img):
    ARRAY_WIDTH = 6200.0
    ARRAY_HEIGHT = 6600.0
    SPOT_SPACING = ARRAY_WIDTH/(31+1)
    
    pixelDimX = (SPOT_SPACING*img.shape[1])/(ARRAY_WIDTH)
    pixelDimY = (SPOT_SPACING*img.shape[0])/(ARRAY_HEIGHT)
    return (x-1)*pixelDimX,(y-1)*pixelDimY



def align_dev_human_heart(raw_counts, spot_coords, exp_name):
    EXP_ORDER = ['ST_Sample_4.5-5PCW_1', 'ST_Sample_4.5-5PCW_2', 
                 'ST_Sample_4.5-5PCW_3', 'ST_Sample_4.5-5PCW_4',
                 'ST_Sample_6.5PCW_1', 'ST_Sample_6.5PCW_2', 
                 'ST_Sample_6.5PCW_3', 'ST_Sample_6.5PCW_4',
                 'ST_Sample_6.5PCW_5', 'ST_Sample_6.5PCW_6',
                 'ST_Sample_6.5PCW_7', 'ST_Sample_6.5PCW_8',
                 'ST_Sample_6.5PCW_9', 'ST_Sample_9PCW_1',
                 'ST_Sample_9PCW_2', 'ST_Sample_9PCW_3',
                 'ST_Sample_9PCW_4', 'ST_Sample_9PCW_5',
                 'ST_Sample_9PCW_6']
    EXP_MAP = {key: value for key, value in zip(EXP_ORDER, np.arange(19) + 1)}
    
    # select 
    exp_id = EXP_MAP[exp_name]
    col_mask = [col for col in raw_counts.columns if col.startswith(f'{exp_id}x')]
    raw_counts = raw_counts[col_mask]
    raw_counts = raw_counts.transpose()
    spot_coords.index = [str(exp_id) + 'x' for _ in range(len(spot_coords))] + spot_coords['x'].astype(str) + ['x' for _ in range(len(spot_coords))] + spot_coords['y'].astype(str)
    
    merged = pd.merge(raw_counts, spot_coords, left_index=True, right_index=True, how='inner')
    raw_counts = raw_counts.reindex(merged.index)
    adata = sc.AnnData(raw_counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    return adata, matrix


def align_ST_counts_with_transform(raw_counts_path, transform_path):
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    with open(transform_path) as file:
        aff_transform = np.array(file.read().split(' '))
        aff_transform = aff_transform.reshape((3, 3)).astype(float).T
    xy = np.array([[idx.split('x')[0], idx.split('x')[1], 1] for idx in raw_counts.index]).astype(float)
    xy_aligned = (aff_transform @ xy.T).T
    adata = sc.AnnData(raw_counts)
    matrix = xy_aligned[:, :2]
    return adata, matrix
    
    
    
def raw_counts_to_pixel(raw_counts_df, img):
    spot_coords = []
    for col in raw_counts_df.columns:
        tup = col.split('_')
        x, y = _ST_spot_to_pixel(float(tup[0]), float(tup[1]), img)
        spot_coords.append([x, y])
    return np.array(spot_coords)
    

def read_ST(meta_table_path=None, raw_counts_path=None, img_path=None, spot_coord_path=None, transform_path=None, ADT_data=False, is_GSE144239=False):
    #raw_counts = pd.read_csv(raw_counts_path, sep='\t')
    img, pixel_size_embedded = _load_image(img_path)
    
    if is_GSE144239:
        raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
        spot_coord = pd.read_csv(spot_coord_path, sep='\t')
        spot_coord.index = spot_coord['x'].astype(str) + ['x' for _ in range(len(spot_coord))] + spot_coord['y'].astype(str)
        merged = pd.merge(spot_coord, raw_counts, left_index=True, right_index=True)
        raw_counts = raw_counts.reindex(merged.index)
        adata = sc.AnnData(raw_counts)
        col1 = merged['pixel_x'].values
        col2 = merged['pixel_y'].values
        matrix = (np.vstack((col1, col2))).T
    
    elif ADT_data:
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
        
    
    elif transform_path is not None:
        adata, matrix = align_ST_counts_with_transform(raw_counts_path, transform_path)
    
    # TODO modify logic later on
    elif meta_table_path is not None and raw_counts_path is not None and spot_coord_path is not None:
        # this works for the developing human heart dataset
        spot_coords = pd.read_csv(spot_coord_path, sep='\t')
        raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
        #meta = pd.read_csv(meta_table_path, sep='\t', index_col=0)
        exp_name = os.path.dirname(spot_coord_path).split('/')[-1]
        adata, matrix = align_dev_human_heart(raw_counts, spot_coords, exp_name)
    else:
        if 'Unnamed: 0' in raw_counts.columns:
            raw_counts.index = raw_counts['Unnamed: 0']
            raw_counts = raw_counts.drop(['Unnamed: 0'], axis=1)
        if meta_table_path is not None:
            meta = pd.read_csv(meta_table_path, sep='\t', index_col=0)
            merged = pd.merge(meta, raw_counts, left_index=True, right_index=True, how='inner')
            raw_counts = raw_counts.reindex(merged.index)
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
            adata = sc.AnnData(raw_counts)
    
    adata.obsm['spatial'] = matrix
    
    # TODO get real pixel size
    my_df = pd.DataFrame(adata.obsm['spatial'], adata.to_df().index, columns=['pxl_col_in_fullres', 'pxl_row_in_fullres'])
    my_df['array_row'] = [int(idx.split('x')[0]) for idx in my_df.index]
    my_df['array_col'] = [int(idx.split('x')[1]) for idx in my_df.index]
    
    pixel_size = _find_pixel_size_from_spot_coords(my_df, inter_spot_dist=200)
    _register_downscale_img(adata, img, pixel_size, spot_size=100.)
    
    return adata, img, pixel_size


def _metric_file_do_dict(metric_file_path):
    metrics = pd.read_csv(metric_file_path)
    dict = metrics.to_dict('records')[0]
    return dict
    

def read_any(path):
    if 'visium' in path.lower():
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
        if alignment_path is None and os.path.exists(os.path.join(path, 'spatial')):
            alignment_path = _find_first_file_endswith(os.path.join(path, 'spatial'), 'autoalignment.json')
        if alignment_path is None:
            json_path = _find_first_file_endswith(path, '.json')
            if json_path is not None:
                f = open(json_path)
                dict = json.load(f)
                if 'oligo' in dict:
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
            GSE167096_count_path = _find_first_file_endswith(path, 'symbol.txt')
            
        GSE203165_count_path = None
        if 'Spatial sequencing of Foreign body granuloma' in path:
            GSE203165_count_path = _find_first_file_endswith(path, 'raw_counts.txt')
            
        seurat_h5_path = _find_first_file_endswith(path, 'seurat.h5ad')
        
        if img_filename is None:
            raise Exception(f"Couldn't detect an image in the directory {path}")
        
        metric_file_path = _find_first_file_endswith(path, 'metrics_summary.csv')
            
        adata, tissue_positions_df, img, raw_bc_matrix, dict = read_10x_visium(
            filtered_bc_matrix_path=filtered_feature_path,
            raw_bc_matrix_path=raw_feature_path,
            spatial_coord_path=spatial_coord_path,
            img_path=os.path.join(path, img_filename),
            alignment_file_path=alignment_path,
            mex_path=mex_path,
            raw_count_path=raw_count_path,
            GSE234047_count_path=GSE234047_count_path,
            GSE180128_count_path=GSE180128_count_path,
            GSE167096_count_path=GSE167096_count_path,
            GSE203165_count_path=GSE203165_count_path,
            seurat_h5_path=seurat_h5_path,
            metric_file_path=metric_file_path
        )
        
        
        os.makedirs(os.path.join(path, 'processed'), exist_ok=True)
        
        adata.var["mito"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], inplace=True)
        
        save_10x_visium(
            adata, 
            os.path.join(path, 'processed'),
            img,
            dict,
            h5_path=filtered_feature_path,
            spatial_path=spatial_coord_path,
        )
        
        return adata
        
    elif 'xenium'in path.lower():
        img_filename = _find_biggest_img(path)
                
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
    
    elif 'ST' in path:
        meta_table_path = None
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
            ADT_data = True
        else:
            ADT_data = False
            
        if "Single Cell and Spatial Analysis of Human Squamous Cell Carcinoma [ST]" in path:
            is_GSE144239 = True
        else:
            is_GSE144239 = False
       
        img_path = _find_biggest_img(path)
        adata, img, pixel_size = read_ST(meta_table_path, raw_counts_path, os.path.join(path, img_path), spot_coord_path, transform_path, ADT_data, is_GSE144239)
        
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        os.makedirs(os.path.join(path, 'processed'), exist_ok=True) 
        
        #save_10x_visium(
        #    adata, 
        #    os.path.join(path, 'processed'),
        #    img,
        #    pixel_size,
        #    h5_path=filtered_feature_path,
        #    spatial_path=spatial_coord_path,
        #)
        return adata
    

def cart_dist(start_spot, end_spot):
    d = np.sqrt((start_spot['pxl_col_in_fullres'] - end_spot['pxl_col_in_fullres']) ** 2 \
        + (start_spot['pxl_row_in_fullres'] - end_spot['pxl_row_in_fullres']) ** 2)
    return d
      
      
def _find_pixel_size_from_spot_coords(my_df, inter_spot_dist=100.):
    df = my_df.copy()
    
    
    max_dist_col = 0
    approx_nb = 0
    best_approx = 0
    df = df.sort_values('array_row')
    for index, row in df.iterrows():
        y = row['array_col']
        x = row['array_row']
        if len(df[df['array_row'] == x]) > 1:
            b = df[df['array_row'] == x]['array_col'].idxmax()
            start_spot = row
            end_spot = df.loc[b]
            dist_px = cart_dist(start_spot, end_spot)
            
            dist_col = abs(df.loc[b, 'array_col'] - y) // 2
            
            approx_nb += 1
            
            if dist_col > max_dist_col:
                max_dist_col = dist_col
                best_approx = inter_spot_dist / (dist_px / dist_col)
            if approx_nb > 3:
                break
            
    if approx_nb == 0:
        raise Exception("Couldn't find two spots on the same row")
            
    return best_approx, max_dist_col
      

def _register_downscale_img(adata, img, pixel_size, spot_size=55.):
    TARGET_PIXEL_EDGE = 1000
    print('image size is ', img.shape)
    downscale_factor = TARGET_PIXEL_EDGE / np.max(img.shape)
    downscaled_fullres = imresize(img, downscale_factor)
    
    # register the image
    adata.uns['spatial'] = {}
    adata.uns['spatial']['ST'] = {}
    adata.uns['spatial']['ST']['images'] = {}
    adata.uns['spatial']['ST']['images']['downscaled_fullres'] = downscaled_fullres
    adata.uns['spatial']['ST']['scalefactors'] = {}
    adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres'] = spot_size / pixel_size
    adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef'] = downscale_factor
    
    return downscaled_fullres, downscale_factor
  

def _get_scalefactors(path: str):
    f = open(path)
    d = json.load(f)
    return d
    

def _alignment_file_to_df(path):
    f = open(path)
    data = json.load(f)
    
    return pd.DataFrame(data['oligo'])
 

def _txt_matrix_to_adata(txt_matrix_path):
    matrix = pd.read_csv(txt_matrix_path, sep='\t')
    matrix = matrix.transpose()

    adata = sc.AnnData(matrix)

    return adata


def _find_slide_version(alignment_df: str) -> str:
    highest_nb_match = -1
    version_file_name = None
    for file in os.listdir('./barcode_coords/'):
        barcode_coords = pd.read_csv(os.path.join('./barcode_coords/', file), sep='\t', header=None)
        barcode_coords = barcode_coords.rename(columns={
            0: 'barcode',
            1: 'array_col',
            2: 'array_row'
        })
        barcode_coords['barcode'] += '-1'
        
        # space rangers provided barcode coords are 1 indexed whereas alignment file are 0 indexed
        barcode_coords['array_col'] -= 1
        barcode_coords['array_row'] -= 1

        spatial_aligned = pd.merge(alignment_df, barcode_coords, on=['array_row', 'array_col'], how='inner')
        if len(spatial_aligned) > highest_nb_match:
            highest_nb_match = len(spatial_aligned)
            version_file_name = file
    return version_file_name
            

def _find_alignment_barcodes(alignment_df: str, adata: sc.AnnData) -> pd.DataFrame:
    barcode_coords = pd.read_csv('./barcode_coords/visium-v1_coordinates.txt', sep='\t', header=None)
    barcode_coords = barcode_coords.rename(columns={
        0: 'barcode',
        1: 'array_col',
        2: 'array_row'
    })
    barcode_coords['barcode'] += '-1'
    
    # space rangers provided barcode coords are 1 indexed whereas alignment file are 0 indexed
    barcode_coords['array_col'] -= 1
    barcode_coords['array_row'] -= 1

    spatial_aligned = pd.merge(alignment_df, barcode_coords, on=['array_row', 'array_col'], how='inner')

    spatial_aligned.index = spatial_aligned['barcode']

    spatial_aligned = spatial_aligned[['in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]

    spatial_aligned = spatial_aligned.reindex(adata.obs.index)

    return spatial_aligned


# taken from https://github.com/scverse/anndata/issues/595
def write_10X_h5(adata, file):
    """Writes adata to a 10X-formatted h5 file.
    
    Note that this function is not fully tested and may not work for all cases.
    It will not write the following keys to the h5 file compared to 10X:
    '_all_tag_keys', 'pattern', 'read', 'sequence'

    Args:
        adata (AnnData object): AnnData object to be written.
        file (str): File name to be written to. If no extension is given, '.h5' is appended.

    Raises:
        FileExistsError: If file already exists.

    Returns:
        None
    """
    
    if '.h5' not in file: file = f'{file}.h5'
    #if os.path.exists(file):
    #    raise FileExistsError(f"There already is a file `{file}`.")
    def int_max(x):
        return int(max(np.floor(len(str(int(np.max(x)))) / 4), 1) * 4)
    def str_max(x):
        return max([len(i) for i in x])

    w = h5py.File(file, 'w')
    grp = w.create_group("matrix")
    grp.create_dataset("barcodes", data=np.array(adata.obs_names, dtype=f'|S{str_max(adata.obs_names)}'))
    grp.create_dataset("data", data=np.array(adata.X.data, dtype=f'<i{int_max(adata.X.data)}'))
    ftrs = grp.create_group("features")
    # this group will lack the following keys:
    # '_all_tag_keys', 'feature_type', 'genome', 'id', 'name', 'pattern', 'read', 'sequence'
    if 'feature_types' not in adata.var:
        adata.var['feature_types'] = ['Unspecified' for _ in range(len(adata.var))]   
    ftrs.create_dataset("feature_type", data=np.array(adata.var.feature_types, dtype=f'|S{str_max(adata.var.feature_types)}'))
    if 'genome' not in adata.var:
        adata.var['genome'] = ['Unspecified_genone' for _ in range(len(adata.var))]
    ftrs.create_dataset("genome", data=np.array(adata.var.genome, dtype=f'|S{str_max(adata.var.genome)}'))
    if 'gene_ids' not in adata.var:
        adata.var['gene_ids'] = ['Unspecified_gene_id' for _ in range(len(adata.var))]
    ftrs.create_dataset("id", data=np.array(adata.var.gene_ids, dtype=f'|S{str_max(adata.var.gene_ids)}'))
    ftrs.create_dataset("name", data=np.array(adata.var.index, dtype=f'|S{str_max(adata.var.index)}'))
    grp.create_dataset("indices", data=np.array(adata.X.indices, dtype=f'<i{int_max(adata.X.indices)}'))
    grp.create_dataset("indptr", data=np.array(adata.X.indptr, dtype=f'<i{int_max(adata.X.indptr)}'))
    grp.create_dataset("shape", data=np.array(list(adata.X.shape)[::-1], dtype=f'<i{int_max(adata.X.shape)}'))


def _helper_mex(path, filename):
    # zip if needed
    file = _find_first_file_endswith(path, filename.strip('.gz'))
    dst = os.path.join(path, filename)
    src = _find_first_file_endswith(path, filename)
    if file is not None and src is None:
        f_in = open(file, 'rb')
        f_out = gzip.open(os.path.join(os.path.join(path), filename), 'wb')
        f_out.writelines(f_in)
        f_out.close()
        f_in.close()
    
    if not os.path.exists(dst) and \
            src is not None:
        shutil.copy(src, dst)


def _load_image(img_path):
    unit_to_micrometers = {
        tifffile.RESUNIT.INCH: 25.4,
        tifffile.RESUNIT.CENTIMETER: 1.e4,
        tifffile.RESUNIT.MILLIMETER: 1.e3,
        tifffile.RESUNIT.MICROMETER: 1.,
    }
    pixel_size_embedded = None
    if img_path.endswith('tiff') or img_path.endswith('tif') or img_path.endswith('btf') or img_path.endswith('TIF'):
        img = tifffile.imread(img_path)
        my_img = tifffile.TiffFile(img_path)
        if 'XResolution' in my_img.pages[0].tags and my_img.pages[0].tags['XResolution'].value[0] != 0:
            # result in micrometers per pixel
            factor = unit_to_micrometers[my_img.pages[0].tags['ResolutionUnit'].value]
            pixel_size_embedded = (my_img.pages[0].tags['XResolution'].value[1] / my_img.pages[0].tags['XResolution'].value[0]) * factor
    else:
        img = np.array(Image.open(img_path))
        
    # sometimes the RGB axis are inverted
    if img.shape[0] == 3:
        img = np.transpose(img, axes=(1, 2, 0))
    if np.max(img) > 1000:
        img = img.astype(np.float64)
        img /= 2**16    
    
    return img, pixel_size_embedded


def _align_tissue_positions(
        alignment_file_path, 
        tissue_positions, 
        adata
):
    if alignment_file_path is not None:

        alignment_df = _alignment_file_to_df(alignment_file_path)
        
        if len(alignment_df) > 0:
            alignment_df = alignment_df.rename(columns={
                'row': 'array_row',
                'col': 'array_col',
                'x': 'pxl_col_in_fullres',
                'y': 'pxl_row_in_fullres'
            })
            df_merged = tissue_positions.rename(columns={
                'pxl_col_in_fullres': 'pxl_col_in_fullres_old',
                'pxl_row_in_fullres': 'pxl_row_in_fullres_old'
            })
        
            alignment_df = alignment_df[['array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]
            df_merged = df_merged.merge(alignment_df, on=['array_row', 'array_col'], how='left')
            
            adata.obs = df_merged[df_merged['in_tissue'] == 1]
            
            col1 = adata.obs['pxl_col_in_fullres'].values
            col2 = adata.obs['pxl_row_in_fullres'].values
            matrix = (np.vstack((col1, col2))).T
            
            adata.obsm['spatial'] = matrix 
            
            tissue_positions['pxl_col_in_fullres'] = df_merged['pxl_col_in_fullres']
            tissue_positions['pxl_row_in_fullres'] = df_merged['pxl_row_in_fullres']
            
        else:
            col1 = tissue_positions['pxl_col_in_fullres'].values
            col2 = tissue_positions['pxl_row_in_fullres'].values        
                
    spatial_aligned = tissue_positions.reindex(adata.obs.index)
    return spatial_aligned


def _alignment_file_to_tissue_positions(alignment_file_path, adata):
    alignment_df = _alignment_file_to_df(alignment_file_path)
    alignment_df = alignment_df.rename(columns={
        'tissue': 'in_tissue',
        'row': 'array_row',
        'col': 'array_col',
        'imageX': 'pxl_col_in_fullres',
        'imageY': 'pxl_row_in_fullres'
    })

    spatial_aligned = _find_alignment_barcodes(alignment_df, adata)
    return spatial_aligned


def _raw_count_to_adata(raw_count_path):
    matrix = pd.read_csv(raw_count_path, sep=',')
    matrix.index = matrix['Gene']
    matrix = matrix.transpose().iloc[1:]

    adata = sc.AnnData(matrix)

    return adata


def GSE206391_split_h5(path):
    adata = sc.read_h5ad(path)
    a = 1
    library_ids = np.unique(adata.obs['library_id'])
    for library_id in library_ids:
        old_library_id = library_id
        library_id = "_".join(library_id.split('_')[:2])
        new_adata = adata.copy()
        adata.obs['int_index'] = np.arange(len(adata.obs))
        df = adata.obs[adata.obs['library_id'] == old_library_id]
        
        new_df = adata.to_df().iloc[df['int_index']]
        #new_df.index = [idx[:-2] for idx in new_df.index]
        new_df.index = [idx + '-1' for idx in new_df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        new_adata.obsm['spatial'] = adata.obsm['spatial'][df['int_index'].values.astype(int)]
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'{library_id}_filtered_feature_bc_matrix.h5'))
        
        
def _GSE206391_copy_dir(path):
    for dir in os.listdir(path):
        if dir.endswith('filtered_feature_bc_matrix.h5'):
            whole_path = os.path.join(path, dir)
            if '21L' in dir:
                sample_name = dir.split('_filtered_feature_bc_matrix.h5')[0].split('_')[1]
            else:
                sample_name = dir.split('_filtered_feature_bc_matrix.h5')[0]
            param = f'mv "{whole_path}" "{path}/"*{sample_name}*'
            subprocess.Popen(param, shell=True)

def GSE234047_to_h5(path):
            
    df = pd.read_csv(path)
    df.index = df['barcode']
    columns_drop = ['barcode', 'prediction_celltype', 'Bipolar', 'Cone', 'Endothelial', 'Fibroblast', 'Immune', 'Interneuron', 'Melanocyte', 'Muller.Astrocyte', 'Pericyte.SMC', 'RGC', 'Rod', 'RPE.x', 'Schwann', 'res_ss', 'region', 'tissue', 'percent_CNV', 'image']
    
    df = df.drop(columns_drop, axis=1)
    
    df.index = [s.split('_')[1].split('-')[0] + '-1' for s in df.index]
    
    adata = sc.AnnData(df)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X)
    return adata


def GSE180128_to_h5(path):
    df = pd.read_csv(path)
    #df.index = df['barcode']
    #columns_drop = ['barcode', 'prediction_celltype', 'Bipolar', 'Cone', 'Endothelial', 'Fibroblast', 'Immune', 'Interneuron', 'Melanocyte', 'Muller.Astrocyte', 'Pericyte.SMC', 'RGC', 'Rod', 'RPE.x', 'Schwann', 'res_ss', 'region', 'tissue', 'percent_CNV', 'image']
    
    #df = df.drop(columns_drop, axis=1)
    
    #df.index = [s.split('_')[1].split('-')[0] + '-1' for s in df.index]
    df.index = df['Unnamed: 0']
    df = df.drop(['Unnamed: 0'], axis=1)
    adata = sc.AnnData(df)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X)
    return adata    


def GSE184369_to_h5(path):
    adata = sc.read_10x_mtx(path)
    
    
    df = pd.read_csv(path)
    df.index = df['barcode']
    columns_drop = ['barcode', 'prediction_celltype', 'Bipolar', 'Cone', 'Endothelial', 'Fibroblast', 'Immune', 'Interneuron', 'Melanocyte', 'Muller.Astrocyte', 'Pericyte.SMC', 'RGC', 'Rod', 'RPE.x', 'Schwann', 'res_ss', 'region', 'tissue', 'percent_CNV', 'image']
    
    df = df.drop(columns_drop, axis=1)
    
    df.index = [s.split('_')[1].split('-')[0] + '-1' for s in df.index]
    
    adata = sc.AnnData(df)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X)
    return adata



def GSE167096_to_h5(path):

    matrix = pd.read_csv(path, sep='\t')
    matrix.index = matrix['Symbol']
    matrix = matrix.transpose().iloc[1:]
    
    #matrix = matrix.replace("NaN", np.nan)

    adata = sc.AnnData(matrix)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X.astype(int))
    return adata


def GSE203165_to_adata(path):
    matrix = pd.read_csv(path, sep='\t', index_col=0)
    matrix = matrix.transpose()
    adata = sc.AnnData(matrix)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X.astype(int))
    return adata
    
    
def GSE236787_split_to_h5(path):
    adata = sc.read_10x_h5(path)
    sampleIDS = ['-1', '-2', '-3', '-4']
    for sampleID in sampleIDS:
        my_adata = adata.copy()
        df = my_adata.obs#.reset_index(drop=True)
        df = df.loc[[i for i in df.index.values if i.endswith(sampleID)]]
        new_df = my_adata.to_df().loc[df.index]
        new_df.index = [s.split('-')[0] + '-1' for s in new_df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        new_adata.obs = df
        new_adata.obs.index = new_df.index
        
        
        #new_adata.uns['spatial'] = my_adata.uns['spatial'][sampleID]
        
        #col1 = new_adata.obs['pxl_col_in_fullres'].values
        #col2 = new_adata.obs['pxl_row_in_fullres'].values
        #matrix = (np.vstack((col1, col2))).T
        
        #new_adata.obsm['spatial'] = matrix 
        
        #adatas.append(new_adata)
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'N{sampleID}filtered_feature_bc_matrix.h5'))

    return df


def _plot_center_square(width, height, length, color, text, offset=0):
    margin_x = (width - length) / 2
    margin_y = (height - length) / 2
    plt.text(width // 2, (height // 2) + offset, text, fontsize=12, ha='center', va='center', color=color)
    plt.plot([margin_x, length + margin_x], [margin_y, margin_y], color=color)
    plt.plot([margin_x, length + margin_x], [height - margin_y, height - margin_y], color=color)
    plt.plot([margin_x, margin_x], [margin_y, height - margin_y], color=color)
    plt.plot([length + margin_x, length + margin_x], [margin_y, height - margin_y], color=color)


def _plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, path):
    plt.imshow(downscaled_img)

    length_estimated = (6500. / pixel_size_estimated) * down_fact

    width = downscaled_img.shape[1]
    height = downscaled_img.shape[0]
    
    if pixel_size_embedded is not None:
        length_embedded = (6500. / pixel_size_embedded) * down_fact
        _plot_center_square(width, height, length_embedded, 'red', '6.5m, embedded', offset=50)
        
    _plot_center_square(width, height, length_estimated, 'blue', '6.5m, estimated')
    
    plt.savefig(path)
    plt.close()


def read_10x_visium(
    img_path: str,
    filtered_bc_matrix_path: str = None,
    raw_bc_matrix_path: str = None,
    spatial_coord_path: str = None,
    alignment_file_path: str = None, 
    mex_path: str = None,
    custom_matrix_path: str = None,
    downsample_factor = None,
    raw_count_path: str = None,
    GSE234047_count_path: str = None,
    GSE180128_count_path: str = None,
    GSE167096_count_path: str = None,
    GSE203165_count_path: str = None,
    seurat_h5_path: str = None,
    metric_file_path: str = None,
    meta_dict: dict = {}
):
    print(f'read image from {img_path}')
    raw_bc_matrix = None

    if filtered_bc_matrix_path is not None:
        adata = sc.read_10x_h5(filtered_bc_matrix_path)
    elif mex_path is not None:
        _helper_mex(mex_path, 'barcodes.tsv.gz')
        _helper_mex(mex_path, 'features.tsv.gz')
        _helper_mex(mex_path, 'matrix.mtx.gz')
            
        adata = sc.read_10x_mtx(mex_path)
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
    elif GSE167096_count_path is not None:
        adata = GSE167096_to_h5(GSE167096_count_path)
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

    tissue_positions_path = _find_first_file_endswith(spatial_coord_path, 'tissue_positions.csv', exclude='aligned_tissue_positions.csv')
    tissue_position_list_path = _find_first_file_endswith(spatial_coord_path, 'tissue_positions_list.csv')
    if tissue_positions_path is not None or tissue_position_list_path is not None:
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

    elif alignment_file_path is not None:
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
    
    scalefactors_path = _find_first_file_endswith(spatial_coord_path, 'scalefactors_json.json')
    if scalefactors_path is not None:
        with open(scalefactors_path) as f:
            scalefactors = json.load(f)
        pixel_size, spot_estimate_dist = _find_pixel_size_from_spot_coords(spatial_aligned)
        #pixel_size = 55. / scalefactors['spot_diameter_fullres']
    else:
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
    
    print(f"'pixel_size_um_embedded' is {pixel_size_embedded}")
    print(f"'pixel_size_um_estimated' is {pixel_size} estimated by averaging over {spot_estimate_dist} spots")
    print(f"'spots_under_tissue' is {len(adata.obs)}")
    
    dict = {**meta_dict, **dict}

    return adata, spatial_aligned, img, raw_bc_matrix, dict


def _save_scalefactors(adata: sc.AnnData, path):
    dict = {}
    dict['tissue_downscaled_fullres_scalef'] = adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
    dict['spot_diameter_fullres'] = adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']
    
    with open(path, 'w') as json_file:
        json.dump(dict, json_file)


def save_10x_visium(adata, path, img, dict, h5_path=None, spatial_path=None):
    
    if h5_path is not None:
        shutil.copy(h5_path, os.path.join(path, 'filtered_feature_bc_matrix.h5'))
    else:
        write_10X_h5(adata, os.path.join(path, 'filtered_feature_bc_matrix.h5'))
    
    if spatial_path is not None:
        shutil.copytree(spatial_path, os.path.join(path, 'spatial'), dirs_exist_ok=True)
    else:
        os.makedirs(os.path.join(path, 'spatial'), exist_ok=True)
        _save_scalefactors(adata, os.path.join(path, 'spatial/scalefactors_json.json'))

    df = adata.obs
    tissue_positions = df[['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']]
    
    tissue_positions.to_csv(os.path.join(path, 'spatial/tissue_positions.csv'), index=True, index_label='barcode')
    
    with open(os.path.join(path, 'metrics.json'), 'w') as json_file:
        json.dump(dict, json_file) 
    
    downscaled_img = adata.uns['spatial']['ST']['images']['downscaled_fullres']
    down_fact = adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
    down_img = Image.fromarray(downscaled_img)
    down_img.save(os.path.join(path, 'downscaled_fullres.jpeg'))
    
    pixel_size_embedded = dict['pixel_size_um_embedded']
    pixel_size_estimated = dict['pixel_size_um_estimated']
    
    _plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, os.path.join(path, 'pixel_size_vis.png'))
    
    
    
    write_wsi(img, os.path.join(path, 'aligned_fullres_HE.ome.tif'), dict)


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
            
            
    _register_downscale_img(adata, img, pixel_size)

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