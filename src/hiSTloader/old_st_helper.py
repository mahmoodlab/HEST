import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import cv2
import math
from scipy.sparse import issparse
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import h5py
Image.MAX_IMAGE_PIXELS = 933120000

# sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


def process_st_data(path, num_hvg, hvg_list,
                            min_counts = 5000, max_counts = 35000, 
                            pct_counts_mt = 20, min_cells = 10, 
                            #qc_filter = True,
                            log_norm = True, 
                            clustering = False
                            ):
    # path = "data/6884-AS/10x_analysis_6884-AS/Sample_6884-AS-S3-D-GEX_TGAGTATC-GAACGAGT"
    # If you want to specify the count file and source image path
    adata = sc.read_visium(
        path, 
        count_file='filtered_feature_bc_matrix.h5', 
        source_image_path='tissue_hires_image.png'
    )
    ############################################
    # keep in tumor obs only - turns out this is taken care by the function sc.read_visium above
    # adata = adata[adata.obs['in_tissue']==1]
    ############################################

    # QC: Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
    adata.var_names_make_unique()

    #if qc_filter:
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

    # convert gene names to captical letters
    adata.var_names = [name.upper() for name in list(adata.var_names)]
    adata.var["gene_name"] = adata.var.index.astype("str")

    gene_list = adata.var_names
    sc.pp.filter_genes(adata, min_cells=min_cells) # only keep genes expressed in more than 10 cells
    gene_filtered = list(set(gene_list) - set(adata.var_names))
    print(f"# genes removed: {len(gene_filtered)}")

    
    if hvg_list != []:
        # in this case, if we use geneformer to extract embedding, where it asks for raw count data
        # so we don't do log-normalization here
        # filter adata by gene name
        print(f'run using given hvg list, len : {len(hvg_list)}')
        # print(adata)
        # print(adata.var)
        adata = adata[:, adata.var['gene_name'].isin(hvg_list)]

        if log_norm:
            print('--------Log Normalization--------')
            sc.pp.normalize_total(adata, inplace=True)
            sc.pp.log1p(adata)
    else:
        # Log Normalization
        print('--------Log Normalization--------')
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        # select highly variable genes (hvgs), must log-norm before hvg extraction
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=num_hvg)

    # clustering
    if clustering:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added="clusters")
    
    return adata, gene_filtered


def read_spatial_adata_v1(st_folder_path, img_path, num_hvg, hvg_list, header=None, **kwargs):
    """
    Read the spatial data from 10x Visium platform

    st_folder_path: path to the folder containing the spatial data 
        (e.g data/6884-AS/10x_analysis_6884-AS/Sample_6884-AS-S1-B-GEX_GTTAAACC-CAAGCCGG)
        where h5 file and tissue_positions_list.csv are located

    img_path: path to the H&E image
    """
    
    ### 1. Process the gene expression matrix into adata
    spatial_coord_path = os.path.join(st_folder_path, "spatial/tissue_positions_list.csv")
    adata, gene_filtered = process_st_data(st_folder_path, num_hvg, hvg_list, **kwargs)
    print(adata)

    ### 2. SPATIAL COORDINATES
    spatial = pd.read_csv(spatial_coord_path, sep="," , na_filter=False, index_col=0, header=header)
    # new addition Feb 7 2023
    spatial.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

    
    df_merged = adata.obs.merge(spatial[['pxl_row_in_fullres', 'pxl_col_in_fullres']], 
                            left_index=True, 
                            right_index=True, 
                            how='left')
    # for easy naming in patching functions, while preserving the original names for scanpy
    df_merged['x_array'] = df_merged['array_row']
    df_merged['y_array'] = df_merged['array_col']
    df_merged['x_pixel'] = df_merged['pxl_row_in_fullres']
    df_merged['y_pixel'] = df_merged['pxl_col_in_fullres']

    adata.obs = df_merged

    # # convert gene names to captical letters
    # adata.var_names=[name.upper() for name in list(adata.var_names)]
    # adata.var["gene_name"]=adata.var.index.astype("str")

    ### 3. H&E IMAGE
    img = np.array(Image.open(img_path))
    print(f'Image shape: {img.shape}')

    return adata, img, gene_filtered

def get_matched_dict_from_df(df, st_dir, img_dir):
    '''
    df has two columns, sample_name and img_name
    The returned matched_dict has the sample_name as the key and a tuple of the st_folder_path of sample_name and path of img_name as the value

    Usage:
    st_dir = '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/st_files'
    img_dir = '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/images'

    df_img_name = pd.read_csv('/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/df_img_name.csv')
            sample_name	img_name
    0	GSM5924042_frozen_a_1	a_1 Frozen.ndpi
    1	GSM5924044_frozen_a_15	a_15 Frozen.ndpi
    2	GSM5924045_frozen_a_17	a_17 Frozen.ndpi
    3	GSM5924043_frozen_a_3	a_3 Frozen.ndpi
    ...

    matched_dict = get_matched_dict_from_df(df_img_name, st_dir, img_dir)
    print(matched_dict):
                    {'GSM5924042_frozen_a_1': ('/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/st_files/GSM5924042_frozen_a_1',
                    '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/images/a_1 Frozen.ndpi'),
                    'GSM5924044_frozen_a_15': ('/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/st_files/GSM5924044_frozen_a_15',
                    '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/images/a_15 Frozen.ndpi'),
                    'GSM5924045_frozen_a_17': ('/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/st_files/GSM5924045_frozen_a_17',
                    '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/images/a_17 Frozen.ndpi'),
                    'GSM5924043_frozen_a_3': ('/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/st_files/GSM5924043_frozen_a_3',
                    '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/lymph_23/images/a_3 Frozen.ndpi'),
                    ...
                    }
    '''
    matched_dict = {}
    for index, row in df.iterrows():
        matched_dict[row['sample_name']] = (os.path.join(st_dir, row['sample_name']), os.path.join(img_dir, row['img_name']))
    return matched_dict


def get_matched_dict(samples_path, images_path):
    """
    NOTE: This is an deprecated version specifically for the 6884-AS dataset folder naming structure, 
    use get_matched_dict_from_df instead for more general usage.

    Generate a dictionary of {sample_name: (st_folder_path: img_path)}

    matched_dict: a dictionary of {sample_name: (st_folder_path, img_path)}

    For example:
    matched_dict =  {'Sample_6884-AS-S2-D-GEX_TGCTTAAC-TGTGGGTG': 
                        ('/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/6884-AS/10x_analysis_6884-AS/Sample_6884-AS-S2-D-GEX_TGCTTAAC-TGTGGGTG',
                            '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/6884-AS/Image_Related_Files/Mahmood_6884-AS-S2-D_V12F07-005.tif'),
                    ...
                    }

    """
    # Initialize an empty dictionary to store matched pairs
    matched_dict = {}

    # Navigate through the samples directory
    for filename in os.listdir(samples_path):
        # Check if the filename starts with 'Sample_'
        if filename.startswith('Sample_'):
            # Extract the ID from the filename
            id = filename.split('_')[1].split('-GEX')[0]
            # Add the full path of the sample to the dictionary
            matched_dict[filename] = os.path.join(samples_path, filename)

    # Navigate through the images directory
    for filename in os.listdir(images_path):
        # Check if the filename starts with 'Mahmood_'
        if filename.startswith('Mahmood_'):
            # Extract the ID from the filename
            id = filename.split('_')[1].split('_')[0]
            # Check if this ID is in the matched_dict keys
            for key in matched_dict.keys():
                if id in key:
                    # If so, add the image to the corresponding sample in the dictionary
                    matched_dict[key] = (matched_dict[key], os.path.join(images_path, filename))
    return matched_dict


def get_data_lists(matched_dict, num_hvg = 1000, hvg_list = [], df_filter = None, **kwargs):
    """
    Get adata_list, img_list, hvgs_union
    To get hvgs_union, we need to get the union of hvgs from all the samples then subtract the genes that do not exist in all samples

    matched_dict: a dictionary of {sample_name: (st_folder_path, img_path)}
            {'Sample_6884-AS-S2-D-GEX_TGCTTAAC-TGTGGGTG': 
                        ('/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/6884-AS/10x_analysis_6884-AS/Sample_6884-AS-S2-D-GEX_TGCTTAAC-TGTGGGTG',
                            '/home/mahmoodlab8/workspace/anthony/projects/ST_codebase/data/6884-AS/Image_Related_Files/Mahmood_6884-AS-S2-D_V12F07-005.tif'),
            ...}

    df_filter looks like: # cv_threshold is used to filter out bad patches
    sample_name	min_counts	max_counts	min_cells	pct_counts_mt	cv_threshold
    GSM5924042_frozen_a_1	5000	20000	50	20	210
    GSM5924044_frozen_a_15	5000	35000	50	20	175
    ...

    """
    hvgs_union = []
    gene_filtered_union = []
    n_spots = 0
    adata_list = []
    img_list = []
    sample_names = list(matched_dict.keys())

    for i in range(len(matched_dict)):
        sample_name = sample_names[i]
        st_folder_path, img_path = list(matched_dict.values())[i]
        if df_filter is not None: # use individual filtering
            
            min_counts, max_counts, min_cells, pct_counts_mt, cv_threshold = df_filter[df_filter['sample_name'] == sample_name].to_numpy()[0][1:]
            assert cv_threshold != 0 # 0 means we discard the whole image, which should be done in the filtering matched_dict
            print(f'Using specific filtering for {sample_name}: min_counts: {min_counts}, max_counts: {max_counts}, min_cells: {min_cells}, pct_counts_mt: {pct_counts_mt}')
            adata, img, gene_filtered = read_spatial_adata_v1(st_folder_path, img_path, num_hvg, hvg_list, 
                                                                min_counts=min_counts, max_counts=max_counts, 
                                                                min_cells=min_cells, pct_counts_mt=pct_counts_mt, **kwargs)
            
            adata = drop_bad_indices_in_adata(adata, img, cv_thresh=cv_threshold)
            
        else:
            adata, img, gene_filtered = read_spatial_adata_v1(st_folder_path, img_path, num_hvg, hvg_list, **kwargs)
        
        print(sample_name, st_folder_path, img_path)
        adata_list.append(adata)
        img_list.append(img)

        if not hvg_list: # if given a hvg list, no need to get the hvgs again
            df_hvg = adata.var['highly_variable']
            df_hvg = df_hvg[df_hvg == True]
            _hvgs = df_hvg.index.tolist()
            hvgs_union.extend(_hvgs)
        gene_filtered_union.extend(gene_filtered)

        n_spots += adata.n_obs

    # get the union of hvgs
    if hvg_list:
        hvgs_union = hvg_list
    print(f'Number of hvgs initially: {len(set(hvgs_union))}')
    print(f'Number of genes discarded: {len(set(gene_filtered_union))}')
    hvgs_union = list(set(hvgs_union) - set(gene_filtered_union)) # get unique ones
    print(f'Number of hvgs after filtering genes: {len(hvgs_union)}')
    print(f'Number of spots to use: {n_spots}')

    # loop through all the slide samples, discard genes that do not exist in all samples
    genes_not_exist_all = []
    for a in adata_list: 
        for gene in hvgs_union:
            #if gene not in a.var_names:
            if gene not in a.var['gene_name'].tolist():
                genes_not_exist_all.append(gene)
    print(f'Number of hvgs that do not exist in all samples: {len(set(genes_not_exist_all))}')
    hvgs_union = list(set(hvgs_union) - set(genes_not_exist_all)) # final version of hvgs, used for all samples
    print(f'Number of hvgs after filtering genes: {len(hvgs_union)}')

    return adata_list, img_list, hvgs_union



# here is a function to filter out the bad patches in adata using computer vision pixel threshold
def drop_bad_indices_in_adata(adata, img, cv_thresh=200):
    # drop bad patches from adata, return adata
    # cv_thresh=200, 160 turns out to be a good threshold that can filter 'blue spot', 200 is not enough

    _adata = adata.copy()
    df = _adata.obs[['x_array', 'y_array', 'x_pixel', 'y_pixel']]
    img_patches_interest = extract_image_patches(df, img, 
                                    patch_size=224
                                    )
    # reindex adata.obs first
    _adata.obs.index = range(_adata.shape[0])
    
    bad_patch_indices = identify_bad_patches(img_patches_interest, threshold=cv_thresh) 
  
    print(f'Before dropping bad patches: {_adata.shape}')
    # drop bad indices from adata
    _adata = _adata[~_adata.obs.index.isin(bad_patch_indices)]
    # reindex adata.obs again
    _adata.obs.index = range(_adata.shape[0])
    print(f'After dropping bad patches: {_adata.shape}')
    
    return _adata

def is_bad_patch(img_patch, threshold=200):
    # Convert the image patch to grayscale
    gray_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    # Compute the average pixel value of the grayscale image
    avg_pixel_value = np.mean(gray_patch)
    # Check if the average pixel value exceeds the threshold
    return avg_pixel_value > threshold

def identify_bad_patches(img_patches, threshold=200):
    bad_patch_indices = []
    for idx, img_patch in enumerate(img_patches):
        if is_bad_patch(img_patch, threshold):
            bad_patch_indices.append(idx)
    return bad_patch_indices

def extract_image_patches(df, image, patch_size = 200):
    # df is a pandas DataFrame containing x_pixel and y_pixel columns
    # image is the histology image (e.g., in RGB format)
    # patch_size is the pixel size of the square patch to extract

    patches = []
    patch_half = patch_size // 2

    max_x = image.shape[0]
    max_y = image.shape[1]

    # make sure our image is correct
    assert int(df['x_pixel'].max()) <= max_x and int(df['y_pixel'].max()) <= max_y

    for _, row in df.iterrows():
        x_pixel = int(row['x_pixel'])
        y_pixel = int(row['y_pixel'])
        
        patch = image[max(0, x_pixel - patch_half):min(max_x, x_pixel + patch_half + 1),
                      max(0, y_pixel - patch_half):min(max_y, y_pixel + patch_half + 1)]

        patches.append(patch)

    return patches

def visualize_patches(patches, cols, figsize):
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

def create_folder(folder_path, clean_folder = True):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    else: # remove all files in folder
        if clean_folder:
            for f in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, f))

def save_img_coord_h5(adata, slide, slide_id, save_folder, patch_size = 200, save_pngs = False):

    create_folder(save_folder, clean_folder = False)

    df_coords = adata.obs[['x_pixel', 'y_pixel']]
    patches = extract_image_patches(df_coords, 
                                    slide, 
                                    patch_size = patch_size
                                    )

    # convert to numpy array                                
    patches = np.array(patches)
    coords = df_coords.to_numpy()
    print(f'patches shape: {patches.shape}')
    print(f'coords shape: {coords.shape}')

    # save csv
    csv_folder = f'{save_folder}/csv'
    create_folder(csv_folder, clean_folder = True)

    df_csv = pd.DataFrame([slide_id], columns=['slide_id'])
    df_csv.to_csv(os.path.join(csv_folder, 'slide_id.csv'), index=False)

    if save_pngs:
        png_folder = f'{save_folder}/png'
        create_folder(png_folder, clean_folder = True)

        for i, patch in enumerate(patches):
            if patch.size > 0:
                patch_img = Image.fromarray(patch)
                patch_img.save(os.path.join(png_folder, f"patch_{i}.png"))
            
        print(f'Saved {len(patches)} patches to {save_folder}')

    h5_folder = f'{save_folder}/h5'
    create_folder(h5_folder, clean_folder = True)

    # Create a new HDF5 file
    h5f = h5py.File(os.path.join(h5_folder, f'{slide_id}.h5'), 'w')

    h5f.create_dataset('imgs', data = patches)
    h5f.create_dataset('coords', data = coords)

    # Close the file
    h5f.close()


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

