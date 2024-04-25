from enum import Enum
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

from .autoalign import autoalign_with_fiducials
import seaborn as sns
import spatialdata_io
from spatialdata._io import write_image
import spatialdata_plot
from packaging import version
import subprocess
import concurrent.futures


Image.MAX_IMAGE_PIXELS = 93312000000


class SpotPacking(Enum):
    ORANGE_CRATE_PACKING = 0
    GRID_PACKING = 1
    

def _copy_to_right_subfolder(dataset_title):
    prefix = f'/mnt/sdb1/paul/data/samples/ST/{dataset_title}'
    #paths = os.listdir(prefix)
    meta_path = '/mnt/sdb1/paul/ST H&E datasets - NCBI (3).csv'
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df[meta_df['dataset_title'] == dataset_title]
    subseries = meta_df['subseries'].values
    sample_name = meta_df['sample_id'].values
    for i in range(len(subseries)):
        try:
            param = f'mv "{prefix}/{sample_name[i]}"* "{os.path.join(prefix, subseries[i])}"'
            subprocess.Popen(param, shell=True)
        except Exception:
            pass
   

def _get_path_from_meta_row(row):
    subseries = row['subseries']
    if isinstance(subseries, float):
        subseries = ""
        
    tech = row['st_technology']
    if isinstance(tech, float):
        tech = 'visium'
    elif 'visium' in tech.lower() and ('visium hd' not in tech.lower()):
        tech = 'visium'
    elif 'xenium' in tech.lower():
        tech = 'xenium'
    elif 'spatial transcriptomics' in tech.lower():
        tech = 'ST'
    elif 'visium hd' in tech.lower():
        tech = 'visium-hd'
    else:
        raise Exception(f'unknown tech {tech}')
    path = os.path.join('/mnt/sdb1/paul/data/samples/', tech, row['dataset_title'], subseries)
    return path   


def copy_processed_images_deprecated(dest, meta_df, cp_spatial=True, cp_downscaled=True, cp_fullres=True):
    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        
        try:
            path = _get_path_from_meta_row(row)
        except Exception:
            continue
        path = os.path.join(path, 'processed')
        if isinstance(row['id'], float):
            my_id = row['id']
            raise Exception(f'invalid sample id {my_id}')
        
        path_fullres = os.path.join(path, 'aligned_fullres_HE.ome.tif')
        if not os.path.exists(path_fullres):
            print(f"couldn't copy {path}")
            continue
        print(f"copying {row['id']}")
        if cp_downscaled:
            path_downscaled = os.path.join(path, 'downscaled_fullres.jpeg')
            os.makedirs(os.path.join(dest, 'downscaled'), exist_ok=True)
            path_dest_downscaled = os.path.join(dest, 'downscaled', row['id'] + '_downscaled_fullres.jpeg')
            shutil.copy(path_downscaled, path_dest_downscaled)
        if cp_spatial:
            path_spatial = os.path.join(path, 'spatial_plots.png')
            os.makedirs(os.path.join(dest, 'spatial_plots'), exist_ok=True)
            path_dest_spatial = os.path.join(dest, 'spatial_plots', row['id'] + '_spatial_plots.png')
            shutil.copy(path_spatial, path_dest_spatial)
        
        if cp_fullres:
            os.makedirs(os.path.join(dest, 'fullres'), exist_ok=True)
            path_dest_fullres = os.path.join(dest, 'fullres', row['id'] + '_aligned_fullres_HE.ome.tif')
            shutil.copy(path_fullres, path_dest_fullres)

        
        #path_downscaled = os.path.join(path, 'downscaled_fullres.jpeg')
        #path_dest_downscaled = os.path.join(dest, 'downscaled', row['id'] + '_downscaled_fullres.jpeg')
        

def open_fiftyone():
    dest = '/mnt/sdb1/paul/images'
    dataset = fo.Dataset.from_images_dir("/mnt/sdb1/paul/images")
    session = fo.launch_app(dataset)    
    
    
def create_joined_gene_plots(meta, gene_plot=False):
    # determine common genes
    if gene_plot:
        plot_dir = 'gene_plots'
    else:
        plot_dir = 'gene_bar_plots'
    common_genes = None
    n = len(meta)
    for index, row in meta.iterrows():
        path = _get_path_from_meta_row(row)
        gene_files = np.array(os.listdir(os.path.join(path, 'processed', plot_dir)))
        if common_genes is None:
            common_genes = gene_files
        else:
            common_genes = np.intersect1d(common_genes, gene_files)
            
    my_dir = '/mnt/sdb1/paul/gene_plot_IDC_xenium'
    os.makedirs(my_dir, exist_ok=True)
    for gene in tqdm(common_genes):
        if gene_plot:
            fig, axes = plt.subplots(1, n)
        else:
            fig, axes = plt.subplots(n, 1)
        i = 0
        for index, row in meta.iterrows():
            path = _get_path_from_meta_row(row)
            gene_path = os.path.join(path, 'processed', plot_dir, gene)
            image = Image.open(gene_path)
            axes[i].imshow(image)
            axes[i].axis('off')
            i += 1
        plt.savefig(os.path.join(my_dir, f'{gene}_subplot.png'), bbox_inches='tight', pad_inches=0, dpi=600)
        plt.subplots_adjust(wspace=0.1)
        plt.close()


def split_join_adata_by_col(path, adata_path, col):
    adata = sc.read_h5ad(os.path.join(path, adata_path))
    samples = np.unique(adata.obs[col])
    for sample in samples:
        sample_adata = adata[adata.obs['orig.ident'] == sample]
        try:
            #write_10X_h5(sample_adata, os.path.join(path, f'{sample}.h5'))
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
            #write_10X_h5(sample_adata, os.path.join(path, f'{sample}.h5'))
        except:
            sample_adata.__dict__['_raw'].__dict__['_var'] = sample_adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
            
            
def GSE205707_split_to_h5ad(path):
    #adata = sc.read_h5ad(os.path.join(path, 'ITD1_1202L.h5ad'))
    #write_10X_h5(adata, os.path.join(path, '1202L.h5'))
    split_join_adata_by_col(path, 'aggregate.h5ad', 'orig.ident')
    split_join_adata_by_col(path, '2L_2R_1197L_1203L_599L_600R.h5ad', 'orig.ident')
    
    
def GSE184369_split_to_h5ad(path):
    feature_path = os.path.join(path, 'old/GSE184369_features.txt')
    features = pd.read_csv(feature_path, header=None)
    features[1] = features[0]
    features[0] = ['Unspecified' for _ in range(len(features))]
    features[2] = ['Unspecified' for _ in range(len(features))]
    features.to_csv(os.path.join(path, 'old/new_features.tsv'), sep='\t', index=False, header=False)
    
    mex_path = os.path.join(path, 'mex')
    adata = sc.read_10x_mtx(mex_path, gex_only=False)
    adata.obs['sample'] = [i.split('-')[0] for i in adata.obs.index]
    adata.obs.index = [i.split('-')[1] for i in adata.obs.index]
    samples = np.unique(adata.obs['sample'])
    for sample in samples:
        sample_adata = adata[adata.obs['sample'] == sample]
        try:
            #write_10X_h5(sample_adata, os.path.join(path, f'{sample}.h5'))
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
            #write_10X_h5(sample_adata, os.path.join(path, f'{sample}.h5'))
        except:
            sample_adata.__dict__['_raw'].__dict__['_var'] = sample_adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
            
            
def pixel_size_to_mag(pixel_size):
    if pixel_size <= 0.1:
        return '>60x'
    elif 0.1 < pixel_size and pixel_size <= 0.25:
        return '60x'
    elif 0.25 < pixel_size and pixel_size <= 0.5:
        return '40x'
    elif 0.5 < pixel_size and pixel_size <= 1:
        return '20x'
    elif 1 < pixel_size and pixel_size <= 4:
        return '10x'  
    elif 4 < pixel_size :
        return '<10x'
    
    
def _get_nan(cell, col_name):
    val = cell[col_name]
    if isinstance(cell[col_name], float):
        return ''
    else:
        return val
        

def create_meta_release(meta_df: pd.DataFrame, version: version.Version):
    META_RELEASE_DIR = '/mnt/sdb1/paul/meta_releases'
    
    metric_subset = [
        'pixel_size_um_embedded', 
        'pixel_size_um_estimated', 
        'fullres_px_width',
        'fullres_px_height',
        'spots_under_tissue',
        'inter_spot_dist',
        'spot_diameter'
    ]
    
    for col in metric_subset:
        meta_df[col] = None
        
    meta_df['inter_spot_dist'] = None
    meta_df['spot_diameter'] = None
    meta_df['nb_genes'] = None
    meta_df['image_filename'] = None
    meta_df['magnification'] = None

    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        path = _get_path_from_meta_row(row)
        f = open(os.path.join(path, 'processed', 'metrics.json'))
        metrics = json.load(f)
        for col in metric_subset:
            meta_df.loc[index][col] = metrics.get(col)
        meta_df.loc[index]['image_filename'] = _sample_id_to_filename(row['id'])
        meta_df.loc[index]['subseries'] = _get_nan(meta_df.loc[index], 'tissue') + _get_nan(meta_df.loc[index], 'disease_comment') + _get_nan(meta_df.loc[index], 'subseries')
        
        meta_df.loc[index]['magnification'] = pixel_size_to_mag(meta_df.loc[index]['pixel_size_um_estimated'])
        
        #TODO remove
        adata = sc.read_h5ad(os.path.join(path, 'processed', 'aligned_adata.h5ad'))
        
        #if row['st_technology'] == 'Visium':
        #    meta_df.loc[index]['inter_spot_dist'] = 100.
        #    meta_df.loc[index]['spot_diameter'] = 55.
        #elif row['st_technology'] == 'Spatial Transcriptomics':
        #    meta_df.loc[index]['inter_spot_dist'] = 200.
        #    meta_df.loc[index]['spot_diameter'] = 100.
        #    if row['dataset_title'] == "Single Cell and Spatial Analysis of Human Squamous Cell Carcinoma [ST]":
        #        meta_df.loc[index]['inter_spot_dist'] = 110.
        #        meta_df.loc[index]['spot_diameter'] = 150.                
        meta_df.loc[index]['nb_genes'] = len(adata.var_names)
        
        
    version_s = str(version).replace('.', '_')
    release_path = os.path.join(META_RELEASE_DIR, f'HEST_v{version_s}.csv')
    if os.path.exists(release_path):
        raise Exception(f'meta already exists at path {release_path}')
    
    release_col_selection = [
        'dataset_title',
        'id',
        'image_filename',
        'organ',
        'disease_state',
        'oncotree_code',
        'species',
        'st_technology',
        'data_publication_date',
        'license',
        'study_link',
        'download_page_link1',
        'inter_spot_dist',
        'spot_diameter',
        'spots_under_tissue',
        'nb_genes',
        'treatment_comment',
        'pixel_size_um_embedded', 
        'pixel_size_um_estimated', 
        'magnification',
        'fullres_px_width',
        'fullres_px_height',
        'subseries'
    ]
    #release_col_selection += metric_subset
    meta_df = meta_df[release_col_selection]
    meta_df = meta_df[meta_df['pixel_size_um_estimated'].isna() | meta_df['pixel_size_um_estimated'] < 1.15]
    meta_df = meta_df[(meta_df['species'] == 'Mus musculus') | (meta_df['species'] == 'Homo sapiens')]  
    meta_df.to_csv(release_path, index=False)


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


def _find_first_file_endswith(dir, suffix, exclude='', anywhere=False):
    files_dir = os.listdir(dir)
    if anywhere:
        matching = [file for file in files_dir if suffix in file and file != exclude]
    else:
        matching = [file for file in files_dir if file.endswith(suffix) and file != exclude]
    if len(matching) == 0:
        return None
    else:
        return os.path.join(dir, matching[0])
    

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


def write_wsi(img, save_path, meta_dict, use_embedded_size=False):
    pixel_size = meta_dict['pixel_size_um_estimated']
    #pixel_size_embedded = meta_dict['pixel_size_um_embedded']
    #if use_embedded_size:
    #    pixel_size = pixel_size_embedded
    if pixel_size is None:
        pixel_size = meta_dict['pixel_size_um_embedded']
    
    
    with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
        options = dict(
            tile=(256, 256), 
            compression='deflate', 
            #metadata=metadata,
            resolution=(
                1. / (pixel_size * 1e-4),
                1. / (pixel_size * 1e-4),
                'CENTIMETER'
            ),
        )
        tif.write(img, **options)


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



def align_dev_human_heart(raw_counts_path, spot_coord_path, exp_name):
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
    
    spot_coords = pd.read_csv(spot_coord_path, sep='\t')
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    
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
    adata.obsm['spatial'] = matrix
    
    return adata


def align_ST_counts_with_transform(raw_counts_path, transform_path):
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    with open(transform_path) as file:
        aff_transform = np.array(file.read().split(' '))
        aff_transform = aff_transform.reshape((3, 3)).astype(float).T
    xy = np.array([[idx.split('x')[0], idx.split('x')[1], 1] for idx in raw_counts.index]).astype(float)
    xy_aligned = (aff_transform @ xy.T).T
    adata = sc.AnnData(raw_counts)
    matrix = xy_aligned[:, :2]
    adata.obsm['spatial'] = matrix
    
    return adata
    
    
    
def raw_counts_to_pixel(raw_counts_df, img):
    spot_coords = []
    for col in raw_counts_df.columns:
        tup = col.split('_')
        x, y = _ST_spot_to_pixel(float(tup[0]), float(tup[1]), img)
        spot_coords.append([x, y])
    return np.array(spot_coords)


def _metric_file_do_dict(metric_file_path):
    metrics = pd.read_csv(metric_file_path)
    dict = metrics.to_dict('records')[0]
    return dict

    


def align_eval_qual_dataset(raw_counts_path, spot_coord_path):
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    spot_coords = pd.read_csv(spot_coord_path, sep='\t')
    
    spot_coords.index = spot_coords['x'].astype(str) + ['x' for _ in range(len(spot_coords))] + spot_coords['y'].astype(str)
    
    merged = pd.merge(raw_counts, spot_coords, left_index=True, right_index=True, how='inner')
    raw_counts = raw_counts.reindex(merged.index)
    adata = sc.AnnData(raw_counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    adata.obsm['spatial'] = matrix
    
    return adata

def align_her2(path, raw_count_path):
    selection_path = _find_first_file_endswith(path, 'selection.tsv')
    spot_coords = pd.read_csv(selection_path, sep='\t')
    raw_counts = pd.read_csv(raw_count_path, sep='\t', index_col=0)
    
    spot_coords.index = spot_coords['x'].astype(str) + ['x' for _ in range(len(spot_coords))] + spot_coords['y'].astype(str)
    
    merged = pd.merge(raw_counts, spot_coords, left_index=True, right_index=True, how='inner')
    raw_counts = raw_counts.reindex(merged.index)
    adata = sc.AnnData(raw_counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    adata.obsm['spatial'] = matrix
    
    return adata
    

def cart_dist(start_spot, end_spot):
    d = np.sqrt((start_spot['pxl_col_in_fullres'] - end_spot['pxl_col_in_fullres']) ** 2 \
        + (start_spot['pxl_row_in_fullres'] - end_spot['pxl_row_in_fullres']) ** 2)
    return d
      
      
def _find_pixel_size_from_spot_coords(my_df, inter_spot_dist=100., packing: SpotPacking = SpotPacking.ORANGE_CRATE_PACKING):
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
            
            div = 1 if packing == SpotPacking.GRID_PACKING else 2
            dist_col = abs(df.loc[b, 'array_col'] - y) // div
            
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
    
    df = pd.DataFrame(data['oligo'])
    
    if 'cytAssistInfo' in data:
        transform = np.array(data['cytAssistInfo']['transformImages'])
        transform = np.linalg.inv(transform)
        matrix = np.column_stack((df['imageX'], df['imageY'], np.ones((len(df['imageX']),))))
        matrix = (transform @ matrix.T).T
        df['imageX'] = matrix[:, 0]
        df['imageY'] = matrix[:, 1]


    return df


def _txt_matrix_to_adata(txt_matrix_path):
    matrix = pd.read_csv(txt_matrix_path, sep='\t')
    matrix = matrix.transpose()

    adata = sc.AnnData(matrix)

    return adata


def _find_slide_version(alignment_df: str, adata: sc.AnnData) -> str:
    highest_nb_match = -1
    version_file_name = None
    barcode_dir = './barcode_coords/'
    for barcode_path in os.listdir(barcode_dir):
        spatial_aligned = _find_alignment_barcodes(alignment_df, os.path.join(barcode_dir, barcode_path))
        nb_match = len(pd.merge(spatial_aligned, adata.obs, left_index=True, right_index=True))
        if nb_match > highest_nb_match:
            highest_nb_match = nb_match
            match_spatial_aligned = spatial_aligned
        #if len(spatial_aligned[spatial_aligned.index.isin(adata.obs.index)]) > highest_nb_match:
    
    if highest_nb_match == 0:
        raise Exception(f"Couldn't find a visium having the following spot barcodes: {adata.obs.index}")
        
    spatial_aligned = match_spatial_aligned.reindex(adata.obs.index)
    return spatial_aligned
            

def _sample_id_to_filename(id):
    return id + '.tif'            

            
def _process_row(dest, row, cp_downscaled, cp_spatial, cp_pyramidal, cp_pixel_vis, cp_adata):
    try:
        path = _get_path_from_meta_row(row)
    except Exception:
        print(f'error with path {path}')
        return
    path = os.path.join(path, 'processed')
    if isinstance(row['id'], float):
        my_id = row['id']
        raise Exception(f'invalid sample id {my_id}')
    
    path_fullres = os.path.join(path, 'aligned_fullres_HE.tif')
    if not os.path.exists(path_fullres):
        print(f"couldn't find {path}")
        return
    print(f"create pyramidal tiff for {row['id']}")
    if cp_pyramidal:
        dst = os.path.join(dest, 'pyramidal', _sample_id_to_filename(row['id']))
        #vips_pyr_cmd = f'LD_LIBRARY_PATH="/mnt/sdb1/paul/vips-8.15.2/mybuild/lib/x86_64-linux-gnu/" /mnt/sdb1/paul/vips-8.15.2/mybuild/bin/vips tiffsave "{path_fullres}" "{dst}" --pyramid --tile --tile-width=256 --tile-height=256 --compression=deflate --bigtiff --subifd'
        bigtiff_option = '' if isinstance(row['bigtiff'], float) or not row['bigtiff']  else '--bigtiff'
        vips_pyr_cmd = f'vips tiffsave "{path_fullres}" "{dst}" --pyramid --tile --tile-width=256 --tile-height=256 --compression=deflate {bigtiff_option}'
        subprocess.call(vips_pyr_cmd, shell=True)
    if cp_downscaled:
        path_downscaled = os.path.join(path, 'downscaled_fullres.jpeg')
        os.makedirs(os.path.join(dest, 'downscaled'), exist_ok=True)
        path_dest_downscaled = os.path.join(dest, 'downscaled', row['id'] + '_downscaled_fullres.jpeg')
        shutil.copy(path_downscaled, path_dest_downscaled)
    if cp_spatial:
        path_spatial = os.path.join(path, 'spatial_plots.png')
        os.makedirs(os.path.join(dest, 'spatial_plots'), exist_ok=True)
        path_dest_spatial = os.path.join(dest, 'spatial_plots', row['id'] + '_spatial_plots.png')
        shutil.copy(path_spatial, path_dest_spatial)
    if cp_pixel_vis:
        path_pixel_vis = os.path.join(path, 'pixel_size_vis.png')
        os.makedirs(os.path.join(dest, 'pixel_vis'), exist_ok=True)
        path_dest_pixel_vis = os.path.join(dest, 'pixel_vis', row['id'] + '_pixel_size_vis.png')
        if not os.path.exists(path_pixel_vis):
            print(f"couldn't find {path_pixel_vis}")
        else:
            shutil.copy(path_pixel_vis, path_dest_pixel_vis)
    if cp_adata:
        path_adata = os.path.join(path, 'aligned_adata.h5ad')
        os.makedirs(os.path.join(dest, 'adata'), exist_ok=True)
        path_dest_adata = os.path.join(dest, 'adata', row['id'] + '.h5ad')
        shutil.copy(path_adata, path_dest_adata)        
        
            
def copy_processed_images(dest: str, meta_df: pd.DataFrame, cp_spatial=True, cp_downscaled=True, cp_pyramidal=True, cp_pixel_vis=True, cp_adata=True, n_job=6):
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_job) as executor:
        # Submit tasks to the executor
        future_results = [executor.submit(_process_row, dest, row, cp_downscaled, cp_spatial, cp_pyramidal, cp_pixel_vis, cp_adata) for _, row in meta_df.iterrows()]

        # Retrieve results as they complete
        for future in concurrent.futures.as_completed(future_results):
            result = future.result()
            print(result)  # Example: Print the processed result
    
    
    #for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    #     _process_row(dest, row, cp_downscaled, cp_spatial, cp_pyramidal, cp_pixel_vis)
    
    
    #copy_processed_images(dest, meta_df, cp_spatial=True, cp_downscaled=True, cp_fullres=True)
            

def _find_alignment_barcodes(alignment_df: str, barcode_path: str) -> pd.DataFrame:
    barcode_coords = pd.read_csv(barcode_path, sep='\t', header=None)
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
    if not isinstance(adata.X, sparse._csc.csc_matrix):
        adata.X = sparse.csr_matrix(adata.X)
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
        tifffile.RESUNIT.NONE: 1.
    }
    pixel_size_embedded = None
    if img_path.endswith('tiff') or img_path.endswith('tif') or img_path.endswith('btf') or img_path.endswith('TIF'):
        img = tifffile.imread(img_path)
            
        
        my_img = tifffile.TiffFile(img_path)
        
        if 'XResolution' in my_img.pages[0].tags and my_img.pages[0].tags['XResolution'].value[0] != 0 and 'ResolutionUnit' in my_img.pages[0].tags:
            # result in micrometers per pixel
            factor = unit_to_micrometers[my_img.pages[0].tags['ResolutionUnit'].value]
            print('ResolutionUnit: ', my_img.pages[0].tags['ResolutionUnit'].value)
            pixel_size_embedded = (my_img.pages[0].tags['XResolution'].value[1] / my_img.pages[0].tags['XResolution'].value[0]) * factor
    else:
        img = np.array(Image.open(img_path))
        
    # sometimes the RGB axis are inverted
    if img.shape[0] == 3:
        img = np.transpose(img, axes=(1, 2, 0))
    elif img.shape[2] == 4: # RGBA to RGB
        img = img[:,:,:3]
    if np.max(img) > 1000:
        img = img.astype(np.float32)
        img /= 2**8
        img = img.astype(np.uint8)
    
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
                'imageX': 'pxl_col_in_fullres', # TODO had a problem here for the prostate dataset
                'imageY': 'pxl_row_in_fullres'
            })
            tissue_positions = tissue_positions.rename(columns={
                'pxl_col_in_fullres': 'pxl_col_in_fullres_old',
                'pxl_row_in_fullres': 'pxl_row_in_fullres_old'
            })
        
            alignment_df = alignment_df[['array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]
            tissue_positions['barcode'] = tissue_positions.index
            df_merged = alignment_df.merge(tissue_positions, on=['array_row', 'array_col'], how='inner')
            df_merged.index = df_merged['barcode']
            df_merged = df_merged.drop('barcode', axis=1)
            df_merged = df_merged.loc[adata.obs.index]
            df_merged = df_merged.reindex(adata.obs.index)
            
            col1 = df_merged['pxl_col_in_fullres'].values
            col2 = df_merged['pxl_row_in_fullres'].values
            matrix = (np.vstack((col1, col2))).T
            
            adata.obsm['spatial'] = matrix 
            
            spatial_aligned = df_merged
            
        else:
            col1 = tissue_positions['pxl_col_in_fullres'].values
            col2 = tissue_positions['pxl_row_in_fullres'].values        
                
            spatial_aligned = tissue_positions.reindex(adata.obs.index)
    else:
        spatial_aligned = tissue_positions
        spatial_aligned = spatial_aligned.loc[adata.obs.index]
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
    alignment_df['in_tissue'] = [True for _ in range(len(alignment_df))]

    spatial_aligned = _find_slide_version(alignment_df, adata)
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


def GSE184384_to_h5(path):
    
    mex_path = _find_first_file_endswith(path, 'mex')
    adata = sc.read_10x_mtx(mex_path)
    
    
            
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



def GSE167096_to_adata(path):
    symbol_path = _find_first_file_endswith(path, 'symbol.txt')

    matrix = pd.read_csv(symbol_path, sep='\t')
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


def infer_row_col_from_barcodes(barcodes_df, adata):
    barcode_path = './barcode_coords/visium-v1_coordinates.txt'
    barcode_coords = pd.read_csv(barcode_path, sep='\t', header=None)
    barcode_coords = barcode_coords.rename(columns={
        0: 'barcode',
        1: 'array_col',
        2: 'array_row'
    })
    barcode_coords['barcode'] += '-1'
    
    # space rangers provided barcode coords are 1 indexed whereas alignment file are 0 indexed
    barcode_coords['array_col'] -= 1
    barcode_coords['array_row'] -= 1
    
    barcodes_df['barcode'] = barcodes_df.index

    spatial_aligned = pd.merge(barcodes_df, barcode_coords, on='barcode', how='inner')

    spatial_aligned.index = spatial_aligned['barcode']

    spatial_aligned = spatial_aligned[['array_row', 'array_col']]

    spatial_aligned = spatial_aligned.reindex(adata.obs.index)
    return spatial_aligned


def GSE217828_to_custom(path):
    raw_counts_path = _find_first_file_endswith(path, 'raw_count.csv')
    raw_counts = pd.read_csv(raw_counts_path)
    raw_counts = raw_counts.transpose()
    raw_counts.index = [idx.split('_')[1].replace('.', '-') for idx in raw_counts.index]
    meta_path = _find_first_file_endswith(path, 'meta_data.csv')
    meta = pd.read_csv(meta_path)
    meta.index = [idx.split('_')[1] for idx in meta.index]
    #meta.index = meta['nCount_SCT']
    matrix = np.column_stack((meta['Coord_x_slide'], abs(meta['Coord_y_slide'])))
    
    
    raw_counts = raw_counts.reindex(meta.index)
    
    adata = sc.AnnData(raw_counts)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X.astype(int))
    
    adata.obsm['spatial'] = matrix
    #adata.obs['']
    
    # TODO infer row and col from template
    spatial_aligned = infer_row_col_from_barcodes(adata.obs, adata)
    adata.obs['array_row'] = spatial_aligned['array_row']
    adata.obs['array_col'] = spatial_aligned['array_col']
    adata.obs['pxl_col_in_fullres'] = matrix[:, 1]
    adata.obs['pxl_row_in_fullres'] = matrix[:, 0]
    adata.obs['in_tissue'] = [True for _ in range(len(adata.obs))]
    
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
    if length > width * 4 and length > height * 4:
        return
    margin_x = (width - length) / 2
    margin_y = (height - length) / 2
    plt.text(width // 2, (height // 2) + offset, text, fontsize=12, ha='center', va='center', color=color)
    plt.plot([margin_x, length + margin_x], [margin_y, margin_y], color=color)
    plt.plot([margin_x, length + margin_x], [height - margin_y, height - margin_y], color=color)
    plt.plot([margin_x, margin_x], [margin_y, height - margin_y], color=color)
    plt.plot([length + margin_x, length + margin_x], [margin_y, height - margin_y], color=color)


def _plot_verify_pixel_size(downscaled_img, down_fact, pixel_size_embedded, pixel_size_estimated, path):
    plt.imshow(downscaled_img)

    width = downscaled_img.shape[1]
    height = downscaled_img.shape[0]
    
    if pixel_size_embedded is not None:
        length_embedded = (6500. / pixel_size_embedded) * down_fact
        _plot_center_square(width, height, length_embedded, 'red', '6.5m, embedded', offset=50)
        
    if pixel_size_estimated is not None:
        length_estimated = (6500. / pixel_size_estimated) * down_fact
        _plot_center_square(width, height, length_estimated, 'blue', '6.5m, estimated')
    
    plt.savefig(path)
    plt.close()



def _save_scalefactors(adata: sc.AnnData, path):
    dict = {}
    dict['tissue_downscaled_fullres_scalef'] = adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
    dict['spot_diameter_fullres'] = adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']
    
    with open(path, 'w') as json_file:
        json.dump(dict, json_file)


def xenium_to_pseudo_visium(df: pd.DataFrame, pixel_size_he):
    y_max = df['y_location'].max()
    y_min = df['y_location'].min()
    x_max = df['x_location'].max()
    x_min = df['x_location'].min()
    
    m = math.ceil((y_max - y_min) / (100 / pixel_size_he))
    n = math.ceil((x_max - x_min) / (100 / pixel_size_he))
    
    features = df['feature_name'].unique()
    
    spot_grid = pd.DataFrame(0, index=range(m * n), columns=features)
    #spot_grid = pd.DataFrame(0, index=range(m * n), columns=features)
    
    # a is the row and b is the column in the pseudo visium grid
    a = np.floor((df['x_location'] - x_min) / (100. / pixel_size_he)).astype(int)
    b = np.floor((df['y_location'] - y_min) / (100. / pixel_size_he)).astype(int)
    
    c = b * n + a
    features = df['feature_name']
    
    cols = spot_grid.columns.get_indexer(features)
    
    spot_grid_np = spot_grid.values.astype(np.uint16)
    #spot_grid_np[c, cols] += 1
    np.add.at(spot_grid_np, (c, cols), 1)
    
    
    if isinstance(spot_grid.columns.values[0], bytes):
        spot_grid.columns = [i.decode('utf-8') for i in spot_grid.columns]
    

    expression_df = pd.DataFrame(spot_grid_np, columns=spot_grid.columns)
    
    coord_df = expression_df.copy()
    coord_df['x'] = x_min + (coord_df.index % n) * (100. / pixel_size_he) + (50. / pixel_size_he)
    coord_df['y'] = y_min + np.floor(coord_df.index / n) * (100. / pixel_size_he) + (50. / pixel_size_he)
    coord_df = coord_df[['x', 'y']]
    
    expression_df.index = [str(i) for i in expression_df.index]
    
    adata = sc.AnnData(expression_df)
    adata.obsm['spatial'] = coord_df[['x', 'y']].values
    
    
    return adata


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