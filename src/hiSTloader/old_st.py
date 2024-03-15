import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings("ignore")
from tqdm import tqdm
from PIL import Image
import openslide
import random
import h5py
import cv2
from kwimage.im_cv2 import imresize

from .old_st_helper import *
from .helpers import read_any

Image.MAX_IMAGE_PIXELS = 933120000
# sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1 #3


def plot_unprocessed(st_dir, img_dir, df_img_name, img_save_dir):
    """
    Here we do not do any processing on the adata/images, 
    we just plot the distribution of gene expression transcripts and spatial overlays on H&E images
    to get a sense of the quality of the data and the gene filter cut-off values (min_counts, max_counts, min_cells)
    """

    print("Plotting plots on unprocessed adata...")
    os.makedirs(img_save_dir, exist_ok=True)
    
    matched_dict = get_matched_dict_from_df(df_img_name, st_dir, img_dir)
    print(f'Number of samples: {len(matched_dict)}')
    sample_names = list(matched_dict.keys())
    st_path_list = [val[0] for val in matched_dict.values()]

    adata_shape_l = []
    for i, path in enumerate(st_path_list):
        adata = sc.read_visium(
            path, 
            count_file='filtered_feature_bc_matrix.h5', 
            source_image_path='tissue_hires_image.png'
        )
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        adata_shape_l.append(f'{adata.shape}[0] x {adata.shape}[1]')
        print(adata)

        ### 1. Plot Distribution ###
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
        # Plot total counts with a threshold
        sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 35000], kde=False, bins=40, ax=axs[1])
        # Plot number of genes by counts for all
        sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
        # Plot number of genes by counts with a threshold
        sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 6000], kde=False, bins=60, ax=axs[3])
        
        # Adding titles to the leftmost subplots for each sample
        axs[0].set_ylabel(f"Sample {i}\n{sample_names[i]}", rotation=0, labelpad=100, size='large', verticalalignment='center')

        plt.tight_layout()
        plt.savefig(os.path.join(img_save_dir, f"{i}_{sample_names[i]}_dist_plots.png"))
        plt.close()
        
        ### 2. Plot Spatial overlays on H&E Images ######
 
        # Generate spatial plots without showing them
        sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts", "pct_counts_in_top_200_genes"],
                    ncols=3, cmap='plasma', alpha_img=0.5, 
                    title=[f"Sample {i}, {sample_names[i]}, total_counts", "n_genes_by_counts", "pct_counts_in_top_200_genes"], show=False)
        
        # Adjust the layout to make room for the custom titles
        plt.tight_layout()
        # Save the figure
        plt.savefig(os.path.join(img_save_dir, f"{i}_{sample_names[i]}_spatial_plots.png"))
        plt.close()

    print(f"Distribution plots saved in {img_save_dir}")
    return adata_shape_l
        

"""
def load_data(root_dir):

    path_list = [] 
    for root, d_names, f_names in os.walk(root_dir):
        if len(f_names) == 0:
            continue
        path_list.append(root)

    print("Loading data...")


    print(f'Number of samples before filtering: {len(path_list)}')

    # use a dataframe to store the individaul filtering parameters: sample_name, min_counts, max_counts, min_cells, pct_counts_mt for each sample
    # here we only do basic filtering, more advanced filtering can be done in the future
    df_filter = pd.DataFrame({'sample_name': list(matched_dict.keys()), 
                            'min_counts': [5000]*len(matched_dict), 
                            'max_counts': [35000]*len(matched_dict), 
                            'min_cells': [50]*len(matched_dict), 
                            'pct_counts_mt': [20]*len(matched_dict),
                            'cv_threshold':[210]*len(matched_dict) # cv_threshold is used for computer vision filtering to discard largely empty images
                            })

    adata_list, img_list, hvgs_union = get_data_lists(matched_dict, 
                                                        hvg_list = [], 
                                                        num_hvg = 800, 
                                                        df_filter = df_filter,
                                                        log_norm = False, 
                                                        # min_counts = 5000, max_counts = 35000, 
                                                        # min_cells = 50,
                                                        # pct_counts_mt = 20, 
                                                        )
    print("Loading data complete")
    return adata_list, img_list, hvgs_union, sample_names, matched_dict"""

###############################################################
#### 1. Plot Distribution of Gene Expression Transcripts ######
# ax1: total_counts
# ax2: n_genes_by_counts
###############################################################
# This can tell the optimal cutoffs for filtering out bad cells (min_counts, max_counts, min_cells)
# and if any abnormal samples need to be removed


def plot_dist_plots(adata_list, sample_names, img_save_dir, start, end):
    print("Plotting distribution plots...")
    os.makedirs(img_save_dir, exist_ok=True)

    # Creating sub-lists of 5 samples each
    for i in range(start, end, 5):
        # Slice the adata_list to get sub-lists of 5 elements each
        sub_list = adata_list[i:i+5]

        # Create a large figure to hold all subplots
        fig, axs = plt.subplots(len(sub_list), 2, figsize=(12, 5*len(sub_list)))  # Adjust the size as needed

        for adata_index, adata in enumerate(sub_list):
            if len(axs.shape) > 1:
                axs = axs[adata_index]
            # Plot total counts for all
            sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
            # Plot total counts with a threshold
            #sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 35000], kde=False, bins=40, ax=axs[adata_index, 1])
            # Plot number of genes by counts for all
            sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[1])
            # Plot number of genes by counts with a threshold
            #sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 6000], kde=False, bins=60, ax=axs[adata_index, 3])
            
            # Adding titles to the leftmost subplots for each sample
            axs[0].set_ylabel(f"Sample { i + adata_index + 1}\n{sample_names[i + adata_index]}", rotation=0, labelpad=100, size='large', verticalalignment='center')

        # Adjust layout for better appearance
        plt.tight_layout()
        # Save the entire figure
        plt.savefig(os.path.join(img_save_dir, f"{i}_{i+len(sub_list)-1}_combined_samples_dist_plots.png"))
        plt.close()  # Close the plot to free memory
    print(f"Distribution plots saved in {img_save_dir}")


"""def generate_metrics_df(path_metrics, load_img = True, cv_thresh = 200, 
                        check_wsi = False,
                        save_path = None):
    
    #all_dirs = True
    #for dir in os.listdir(path):
    #    if not os.path.isdir(os.path.join(path, dir)):
    #        all_dirs = False
        
    path_list = []    
    for root, d_names, f_names in os.walk(path_metrics):
        if len(f_names) == 0:
            continue
        path_list.append(root)
        
    #print(f'Number of samples: {len(matched_dict)}')
    #sample_names = list(matched_dict.keys())
    #st_path_list = [val[0] for val in matched_dict.values()]

    df_spatial_l = []
    adata_list = []
    img_list = []
    wsi_list = []
    patches_list = []

    # metrics
    adata_shape_l = []
    pix_size_l = []
    resolution_l = []
    mag_l = []
    img_shape_l = []
    wsi_level0_l = []
    samples_read_l = []

    # pixel coords
    bottom_x_y_l = []
    bottom_x_y_pixel_l = []
    top_x_y_l = []
    top_x_y_pixel_l = []
    max_pixel_l = []
    min_pixel_l = []

    # patch metrics
    bad_patch_percentage_l = []
    good_patch_num_l = []
    bad_patch_num_l = []
    cv_thres_l = []

    for path in path_list:
        # print(key)
        #samples_read_l.append(key)
        adata, img = read_any(path)
        
        #sc.pp.calculate_qc_metrics(adata, inplace=True)
        pix_size, resolution, magnification = get_spot_pixel_size_resolution(adata)
        pix_size_l.append(pix_size)
        resolution_l.append(resolution)
        mag_l.append(magnification)
        adata_shape_l.append(f'{adata.shape[0]} x {adata.shape[1]}')

        #spatial_coord_path = os.path.join(st_path, "spatial/tissue_positions_list.csv")
        #spatial = pd.read_csv(spatial_coord_path, sep="," , na_filter=False, index_col=0, header=None)
        #spatial.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
        #df_merged = adata.obs.merge(spatial[['pxl_row_in_fullres', 'pxl_col_in_fullres']], 
        #                        left_index=True, 
        #                        right_index=True, 
        #                        how='left')
        # for easy naming in patching functions, while preserving the original names for scanpy
        df_merged['x_array'] = df_merged['array_row']
        df_merged['y_array'] = df_merged['array_col']
        df_merged['x_pixel'] = df_merged['pxl_row_in_fullres']
        df_merged['y_pixel'] = df_merged['pxl_col_in_fullres']

        adata.obs = df_merged
        #print(adata)
        df_spatial_l.append(df_merged)
        adata_list.append(adata)

        # get the spot coordinate at top/bottom of the image and the pixel coordinate, use array_col to locate
        df_merged = df_merged[['array_col', 'array_row', 'pxl_row_in_fullres', 'pxl_col_in_fullres']]
        df_merged = df_merged.sort_values(by=['array_col', 'array_row'])
        # get the first spot when array_col = min
        y, x, x_pixel, y_pixel = df_merged[df_merged['array_col'] == df_merged['array_col'].min()].iloc[0,:4].values
        bottom_x_y_l.append(f'{x} , {y}')
        bottom_x_y_pixel_l.append(f'{x_pixel} , {y_pixel}')

        y, x, x_pixel, y_pixel = df_merged[df_merged['array_col'] == df_merged['array_col'].max()].iloc[0,:4].values
        top_x_y_l.append(f'{x} , {y}')
        top_x_y_pixel_l.append(f'{x_pixel} , {y_pixel}')

        min_pixel_l.append(f'{df_merged["pxl_row_in_fullres"].min()} , {df_merged["pxl_col_in_fullres"].min()}')
        max_pixel_l.append(f'{df_merged["pxl_row_in_fullres"].max()} , {df_merged["pxl_col_in_fullres"].max()}')

        # # convert gene names to captical letters
        # adata.var_names=[name.upper() for name in list(adata.var_names)]
        # adata.var["gene_name"]=adata.var.index.astype("str")

        ### 3. H&E IMAGE
        if load_img:
            img = np.array(Image.open(img_path))
            #print(f'Image shape: {img.shape}')
            img_list.append(img)
            img_shape_l.append(f'{img.shape[0]} x {img.shape[1]} x {img.shape[2]}')
            if check_wsi:
                wsi = openslide.OpenSlide(img_path)
                #print(wsi.level_dimensions)
                wsi_level0_l.append(f'{wsi.level_dimensions[0][0]} x {wsi.level_dimensions[0][1]}')
                wsi_list.append(wsi)
            else:
                wsi_level0_l.append('None')
                wsi_list.append('None')
            
            df = adata.obs[['x_array', 'y_array', 'x_pixel', 'y_pixel']]
            patch_pixel_size = int(pix_size+100) # 20x mag: 142.04 + 100
            try:
                cv_thresh = 200 # 160 turns out to be a good threshold that can filter 'blue spot', 200 is not enough
                img_patches_interest = extract_image_patches(df, img, 
                                                    patch_size = patch_pixel_size,
                                                    )
                bad_patch_indices = identify_bad_patches(img_patches_interest, threshold=cv_thresh) 
                bad_patches_l = list(np.array(img_patches_interest)[bad_patch_indices])
                # Convert bad_patch_indices to a set for faster lookup
                bad_patch_indices_set = set(bad_patch_indices)
                # Use a list comprehension to filter out the bad patches
                good_patches_l = [patch for j, patch in enumerate(img_patches_interest) if j not in bad_patch_indices_set]
                
                #print(f'i = {i}, {sample_names[i]}')
                # print(f'Threshold = {cv_thresh}')
                # print(f'# Good patches: {len(good_patches_l)}, # Bad patches: {len(bad_patches_l)}')
                # print(f'Percentage of bad patches: {len(bad_patch_indices)/len(img_patches_interest) * 100:.1f}%')
                cv_thres_l.append(cv_thresh)
                good_patch_num_l.append(len(good_patches_l))
                bad_patch_num_l.append(len(bad_patches_l))
                bad_patch_percentage_l.append(f'{len(bad_patch_indices)/len(img_patches_interest) * 100:.1f}%')
                # assert len(good_patches_l) + len(bad_patches_l) == len(img_patches_interest)
            except:
                print(f'Error in image')
                img_shape_l.append('None')
                wsi_level0_l.append('None')
                img_list.append('None')
                wsi_list.append('None')
                cv_thres_l.append('None')
                good_patch_num_l.append('None')
                bad_patch_num_l.append('None')
                bad_patch_percentage_l.append('None')
                continue

        else:
            img_shape_l.append('None')
            wsi_level0_l.append('None')
            img_list.append('None')
            wsi_list.append('None')
            cv_thres_l.append('None')
            good_patch_num_l.append('None')
            bad_patch_num_l.append('None')
            bad_patch_percentage_l.append('None')

    # make a dataframe of the metrics
    df_metrics = pd.DataFrame({'sample_name' : samples_read_l,
                                'adata_shape': adata_shape_l,
                                'img_shape (height, width, channels)': img_shape_l,
                                'wsi_level_0 (width, height)': wsi_level0_l,
                                'spot_pixel_size': pix_size_l,
                                'resolution (um/pixel)': resolution_l,
                                'magnification': mag_l,
                                'good_patch_num': good_patch_num_l,
                                'bad_patch_num': bad_patch_num_l,
                                'bad_patch_percentage': bad_patch_percentage_l,
                                'cv_threshold': cv_thres_l,
                                'bottom_x_y': bottom_x_y_l,
                                'bottom_x_y_pixel': bottom_x_y_pixel_l,
                                'top_x_y': top_x_y_l,
                                'top_x_y_pixel': top_x_y_pixel_l,
                                'min_pixel_x_y': min_pixel_l,
                                'max_pixel_x_y': max_pixel_l,
                                })
    
    if save_path is not None:
        df_metrics.to_csv(save_path, index=False)
        print(f'Dataframe saved to {save_path}')

    return df_metrics"""
