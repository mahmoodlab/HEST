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


def _find_first_file_endswith(dir, suffix):
    files_dir = os.listdir(dir)
    matching = [file for file in files_dir if file.endswith(suffix)]
    if len(matching) == 0:
        return None
    else:
        return os.path.join(dir, matching[0])


def save_aligned_data(path, adata: sc.AnnData, tissue_positions_df: pd.DataFrame, img, filtered_bc_matrix):
    
    save_path = os.path.join(path, 'processed')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    adata.write_h5ad(os.path.join(save_path, 'aligned_spatial.h5'))
   
    #if os.path.exists(os.path.join(path, 'spatial/tissue_positions_list.csv')):
    #    os.rename(os.path.join(path, 'spatial/tissue_positions_list.csv'), 
    #              os.path.join(path, 'processed/old_tissue_positions_list.csv'))
    #elif os.path.exists(os.path.join(path, 'spatial/tissue_positions.csv')):
    #    os.rename(os.path.join(path, 'spatial/tissue_positions.csv'), 
    #              os.path.join(path, 'processed/old_tissue_positions.csv'))
           
    tissue_positions_df.to_csv(os.path.join(path, 'spatial/aligned_tissue_positions.csv'), index_label='barcode')
    
    with tifffile.TiffWriter(os.path.join(path, 'aligned_fullres_HE.ome.tif'), bigtiff=True) as tif:
        tif.write(img)
        
    exists = _find_first_file_endswith(path, 'filtered_feature_bc_matrix.h5') is not None
    if not exists and filtered_bc_matrix is not None:
        filtered_bc_matrix.write_h5ad(os.path.join(path, 'filtered_feature_bc_matrix.h5'))
   
   
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
 
    
###############################################################
#### 2. Plot Spatial overlays on H&E Images ######
# ax1: total_counts
# ax2: n_genes_by_counts
# ax3: pct_counts_in_top_200_genes
###############################################################

def save_spatial_plot(adata, save_path, name, processed=False):
    print("Plotting spatial plots...")


    #print(sample_name)
    
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
    
        save_aligned_data(path, adata, tissue_positions_df, img, raw_bc_matrix)
        if save_plots:
            save_spatial_plot(adata, os.path.join(path, 'processed'), name)
            #save_spatial_plot(processed_adata, os.path.join(path, 'aligned'), name, processed=True)
        
    
    return adata_list, img_list, sample_names


def _find_biggest_img(path):
    biggest_size = -1
    biggest_img_filename = None
    for file in os.listdir(path):
        if file.endswith('.tif') or file.endswith('.jpg') or file.endswith('.btf') or file.endswith('.png') or file.endswith('.tiff'):
            if file not in ['aligned_fullres_HE.ome.tif', 'morphology.ome.tif', 'morphology_focus.ome.tif', 'morphology_mip.ome.tif']:
                size = os.path.getsize(os.path.join(path, file))
                if size > biggest_size:
                    biggest_img_filename = file
                    biggest_size = size
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
        df = my_adata.obs.reset_index(drop=True)
        df = df[df['sampleID'] == sampleID]
        new_df = my_adata.to_df().loc[df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
    
        new_adata.X = sparse.csr_matrix(new_adata.X)
        new_adata.obs = my_adata.obs[my_adata.obs['sampleID'] == sampleID]
        
        
        new_adata.uns['spatial'] = my_adata.uns['spatial'][sampleID]
        new_adata.obsm['spatial'] = my_adata.obsm['spatial'][df.index]
        #adatas.append(new_adata)
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'{sampleID}_filtered_feature_bc_matrix.h5'))


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
        
            
        adata, tissue_positions_df, img, raw_bc_matrix = read_10x_visium(
            filtered_bc_matrix_path=filtered_feature_path,
            raw_bc_matrix_path=raw_feature_path,
            spatial_coord_path=spatial_coord_path,
            img_path=os.path.join(path, img_filename),
            alignment_file_path=alignment_path,
            mex_path=mex_path
        )
        
        os.makedirs(os.path.join(path, 'processed'), exist_ok=True)
        
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        save_10x_visium(
            adata, 
            os.path.join(path, 'processed'),
            img,
            h5_path=filtered_feature_path,
            spatial_path=spatial_coord_path,
        )
        
        return adata, tissue_positions_df, img, raw_bc_matrix
        
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
      
      
def _find_pixel_size_from_spot_coords(df):
    for index, row in df.iterrows():
        y = row['array_row']
        pxl_x = row['pxl_col_in_fullres']
        pxl_y = row['pxl_row_in_fullres']
        x = row['array_col']
        if len(df[df['array_col'] == x]) > 1:
            b = df[df['array_col'] == x].index.max()
            dist_col = abs(df.loc[b, 'array_row'] - y)
            dist_px_col = abs(df.loc[b, 'pxl_col_in_fullres'] - pxl_x)
            dist_px_row = abs(df.loc[b, 'pxl_row_in_fullres'] - pxl_y)
            dist_px = np.max([dist_px_col, dist_px_row])
            
            return 100 / (dist_px / dist_col)
    raise Exception("Couldn't find two spots on the same row")
      

def _register_downscale_img(adata, img, pixel_size):
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
    adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres'] = 55. / pixel_size
    adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef'] = downscale_factor
  

def _get_scalefactors(path: str):
    f = open(path)
    d = json.load(f)
    return d


def read_custom_1(
    image_path1: str,
    image_path2: str,
    tissue_positions1: str,
    tissue_positions2: str,
    h5_path: str,
):
    adata = sc.read_10x_h5(os.path.join(my_path, 'GSE214989_counts_embryo_visium.h5'))

    img = np.array(Image.open(os.path.join(my_path, 'GSM6619680_220420_sATAC_V10B01-031_B1_NB-Spot000001.jpg')))
    #img3 = np.array(Image.open(os.path.join(my_path, 'GSM6619681_211007_V10S29-086_D1-Spot000001.jpg')))
    #img = tifffile.imread(os.path.join(my_path, 'GSM6619680_220420_sATAC_V10B01-031_B1_NB-Spot000001.jpg'))
    #img4 = imresize(img3, 0.025)
    #plt.imshow(img4)
    #plt.show()
    list1 = pd.read_csv(os.path.join(my_path, 'GSM6619680_220420_sATAC_V10B01-031_B1_tissue_positions_list.csv'), header=None)
    #list2 = pd.read_csv(os.path.join(my_path, 'GSM6619681_211007_V10S29-086_D1_tissue_positions_list.csv'), header=None)
    #list3 = pd.read_csv(os.path.join(my_path, 'GSM6619682_V10B01-135_D1_tissue_positions_list.csv'), header=None)

    list1 = list1.rename(columns={1: "in_tissue", # in_tissue: 1 if spot is captured in tissue region, 0 otherwise
                            2: "array_row", # spot row index
                            3: "array_col", # spot column index
                            4: "pxl_row_in_fullres", # spot x coordinate in image pixel
                            5: "pxl_col_in_fullres"}) # spot y coordinate in image pixel


    img2 = imresize(img, 0.025)

    list1 = list1[list1['in_tissue'] == 1]

    plt.imshow(img2)
    plt.scatter(list1["pxl_col_in_fullres"] * 0.025, list1["pxl_row_in_fullres"] * 0.025, s=5)
    plt.show()   
    

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
    ftrs.create_dataset("feature_types", data=np.array(adata.var.feature_types, dtype=f'|S{str_max(adata.var.feature_types)}'))
    if 'genome' not in adata.var:
        adata.var['genome'] = ['Unspecified_genone' for _ in range(len(adata.var))]
    ftrs.create_dataset("genome", data=np.array(adata.var.genome, dtype=f'|S{str_max(adata.var.genome)}'))
    ftrs.create_dataset("id", data=np.array(adata.var.gene_ids, dtype=f'|S{str_max(adata.var.gene_ids)}'))
    ftrs.create_dataset("name", data=np.array(adata.var.index, dtype=f'|S{str_max(adata.var.index)}'))
    grp.create_dataset("indices", data=np.array(adata.X.indices, dtype=f'<i{int_max(adata.X.indices)}'))
    grp.create_dataset("indptr", data=np.array(adata.X.indptr, dtype=f'<i{int_max(adata.X.indptr)}'))
    grp.create_dataset("shape", data=np.array(list(adata.X.shape)[::-1], dtype=f'<i{int_max(adata.X.shape)}'))


def __helper_mex(path, filename):
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


def read_10x_visium(
    img_path: str,
    filtered_bc_matrix_path: str = None,
    raw_bc_matrix_path: str = None,
    spatial_coord_path: str = None,
    alignment_file_path: str = None, 
    mex_path: str = None,
    custom_matrix_path: str = None,
    downsample_factor = None
):
    """
    Read the spatial data from 10x Visium platform and filter out spots that are not under tissue
    st_gene_expression_path: path to the gene expression matrix (.h5 file)
    spatial_coord_path: path to the spatial coordinates (e.g dataV1_Human_Lymph_Node/spatial/tissue_positions_list.csv)
    img_path: path to the H&E image
    """
    
    raw_bc_matrix = None

    if filtered_bc_matrix_path is not None:
        adata = sc.read_10x_h5(filtered_bc_matrix_path)
    elif mex_path is not None:
        __helper_mex(mex_path, 'barcodes.tsv.gz')
        __helper_mex(mex_path, 'features.tsv.gz')
        __helper_mex(mex_path, 'matrix.mtx.gz')

            
        adata = sc.read_10x_mtx(mex_path)
    elif raw_bc_matrix_path is not None:
        adata = sc.read_10x_h5(raw_bc_matrix_path)
    elif custom_matrix_path is not None:
        adata = _txt_matrix_to_adata(custom_matrix_path)

    adata.var_names_make_unique()
    print(adata)
    
    if adata.obs.index[0][-1] != '-':
        print('append -1 to the barcodes')
        adata.obs.index = [idx + '-1' for idx in adata.obs.index]

    if spatial_coord_path is not None:
        tissue_positions_path = _find_first_file_endswith(spatial_coord_path, 'tissue_positions.csv')
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

        if alignment_file_path is not None:        
            alignment_df = _alignment_file_to_df(alignment_file_path)
            
            if len(alignment_df) > 0:
                alignment_df = alignment_df.rename(columns={
                    'row': 'array_row',
                    'col': 'array_col',
                    'x': 'pxl_col_in_fullres',
                    'y': 'pxl_row_in_fullres'
                    #'imageX': 'pxl_col_in_fullres',
                    #'imageY': 'pxl_row_in_fullres'
                })
                df_merged = tissue_positions.rename(columns={
                    'pxl_col_in_fullres': 'pxl_col_in_fullres_old',
                    'pxl_row_in_fullres': 'pxl_row_in_fullres_old'
                })
            
                alignment_df = alignment_df[['array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']]
                #original_index = df_merged.index
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
        assert np.array_equal(spatial_aligned.index, adata.obs.index)

    elif alignment_file_path is not None:
        alignment_df = _alignment_file_to_df(alignment_file_path)
        #if len(alignment_df) != len(adata.obs):
        #    raise Exception(
        #        "the number of spots don't match between the alignment file and the"
        #        " gene matrix, please provide a tissue_positions.csv/tissue_positions_list.csv"
        #        " to align the barcodes")
        alignment_df = alignment_df.rename(columns={
            'tissue': 'in_tissue',
            'row': 'array_row',
            'col': 'array_col',
            #'x': 'pxl_col_in_fullres',
            #'y': 'pxl_row_in_fullres'
            'imageX': 'pxl_col_in_fullres',
            'imageY': 'pxl_row_in_fullres'
        })

        spatial_aligned = _find_alignment_barcodes(alignment_df, adata)
        
    else:
        raise Exception("a tissue_positions_list.csv/tissue_positions.csv or an alignment path must be provided")

    
    col1 = spatial_aligned['pxl_col_in_fullres'].values
    col2 = spatial_aligned['pxl_row_in_fullres'].values
    
    matrix = np.vstack((col1, col2)).T
    
    adata.obsm['spatial'] = matrix
    
    scalefactors_path = _find_first_file_endswith(spatial_coord_path, 'scalefactors_json.json')
    if scalefactors_path is not None:
        scalefactors = _get_scalefactors(scalefactors_path)
        pixel_size = 55. / scalefactors['spot_diameter_fullres']
    else:
        pixel_size = _find_pixel_size_from_spot_coords(spatial_aligned)
    

    ### More adata processing
    # add into adata.obs
    adata.obs = spatial_aligned
    # filter out spots outside tissue region
    #if in_tissue_only:
    #    adata = adata[adata.obs["in_tissue"] == 1]
        

    if img_path.endswith('tiff') or img_path.endswith('tif') or img_path.endswith('btf'):
        img = tifffile.imread(img_path)
    else:
        img = np.array(Image.open(img_path))
        
    # sometimes the RGB axis are inverted
    if img.shape[0] == 3:
        img = np.transpose(img, axes=(1, 2, 0))
    if np.max(img) > 1000:
        img = img.astype(np.float64)
        img /= 2**16

        
        #affine = Affine(matrix_cyt)
        
        #img2 = warp_affine(img, affine, dsize='auto', large_warp_dim=1000)
        #img = img2
        
        #plt.imshow(img)
        #plt.show()
        
        #f.close()
        
    # register the image
    #pixel_size = 
    _register_downscale_img(adata, img, pixel_size)

    return adata, spatial_aligned, img, raw_bc_matrix


def _save_scalefactors(adata: sc.AnnData, path):
    dict = {}
    dict['tissue_downscaled_fullres_scalef'] = adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
    dict['spot_diameter_fullres'] = adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']
    
    with open(path, 'w') as json_file:
        json.dump(dict, json_file)


def save_10x_visium(adata, path, img, h5_path=None, spatial_path=None):
    
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
    
    
    with tifffile.TiffWriter(os.path.join(path, 'aligned_fullres_HE.ome.tif'), bigtiff=True) as tif:
        tif.write(img)


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