import pandas as pd
from src.hiSTloader.old_st import *
from src.hiSTloader.helpers import process_all, extract_patch_expression_pairs, visualize_patches, extract_image_patches, download_from_meta_df, read_10x_visium
import tifffile
from PIL import Image
from kwimage.im_cv2 import warp_affine, imresize

import json

Image.MAX_IMAGE_PIXELS = 9331200000

def load_from_image_matrix_align(img_path, matrix_path, alignment_path):
    img = np.array(Image.open(img_path))
    adata = sc.read_10x_h5(matrix_path)
    
    file = open(alignment_path)
    tissue = json.load(file)['oligo']
    df = pd.DataFrame(tissue)
    df = df.rename(columns={
        'row': 'array_row',
        'col': 'array_col',
        #'x': 'pxl_col_in_fullres',
        #'y': 'pxl_row_in_fullres'
        'imageX': 'pxl_col_in_fullres',
        'imageY': 'pxl_row_in_fullres'
    })
    
    adata.obsm['spatial'] = np.column_stack((df['pxl_col_in_fullres'].values, df['pxl_row_in_fullres'].values))
    
    a = 3
    

def custom():
    #data_path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Visium CytAssist Gene Expression Libraries of Post-Xenium Human Colon Cancer (FFPE)/Control, Replicate 1'
    data_path = '/mnt/ssd/paul/ST-histology-loader/data/samples/xenium/Human Tonsil Data with Xenium Human Multi-Tissue and Cancer Panel/Reactive follicular hyperplasia'
    my_path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Spatial transcriptomics profiling of the developing mouse embryo/GSM6619680'
    
    adata = sc.read_10x_h5(os.path.join(my_path, 'GSE214989_counts_embryo_visium.h5'))
    
    img = np.array(Image.open(os.path.join(my_path, 'GSM6619680_220420_sATAC_V10B01-031_B1_NB-Spot000001.jpg')))
    #img3 = np.array(Image.open(os.path.join(my_path, 'GSM6619681_211007_V10S29-086_D1-Spot000001.jpg')))
    #img = tifffile.imread(os.path.join(my_path, 'GSM6619680_220420_sATAC_V10B01-031_B1_NB-Spot000001.jpg'))
    #img4 = imresize(img3, 0.025)
    #plt.imshow(img4)
    #plt.show()
    list1 = pd.read_csv(os.path.join(my_path, 'GSM6619680_220420_sATAC_V10B01-031_B1_tissue_positions_list.csv'), header=None)
    list2 = pd.read_csv(os.path.join(my_path, 'GSM6619681_211007_V10S29-086_D1_tissue_positions_list.csv'), header=None)
    #list3 = pd.read_csv(os.path.join(my_path, 'GSM6619682_V10B01-135_D1_tissue_positions_list.csv'), header=None)
    
    list1 = list1.rename(columns={1: "in_tissue", # in_tissue: 1 if spot is captured in tissue region, 0 otherwise
                                2: "array_row", # spot row index
                                3: "array_col", # spot column index
                                4: "pxl_row_in_fullres", # spot x coordinate in image pixel
                                5: "pxl_col_in_fullres"}) # spot y coordinate in image pixel

    list2 = list2.rename(columns={1: "in_tissue", # in_tissue: 1 if spot is captured in tissue region, 0 otherwise
                                2: "array_row", # spot row index
                                3: "array_col", # spot column index
                                4: "pxl_row_in_fullres", # spot x coordinate in image pixel
                                5: "pxl_col_in_fullres"}) # spot y coordinate in image pixel

    list1[0] = list1[0] + '_1'
    list2[0] = list2[0] + '_2'
    
    
    img2 = imresize(img, 0.025)
    
    #list1 = list1[list1['in_tissue'] == 1]
    #list2 = list2[list2['in_tissue'] == 1]
    #list1 = list1[list1['in_tissue'] == 1]
    
    names = adata.obs_names
    
    missing_values = np.setdiff1d(list1[0].values, names.values)
    my_index = np.setdiff1d(list1[0].values, missing_values)
    
    
    adata1 = adata[my_index]
    
    names = adata.obs_names
    
    plt.imshow(img2)
    plt.scatter(list1["pxl_col_in_fullres"] * 0.025, list1["pxl_row_in_fullres"] * 0.025, s=5)
    plt.show()
    
    

def main():
    project = 'test_paul'
    sample_path = '/mnt/ssd/paul/ST-histology-loader/data/samples'
    
   # adata = sc.read_10x_h5('/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Distinct mesenchymal cell states mediate prostate cancer progression [Spatial Transcriptomics]/PRN/GSM7914975_WT_feature_bc_matrix.h5')
    
    #read_any()
    
    load_from_image_matrix_align(
        '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Zika virus co-opts miRNA networks to persist in placental microenvironments detected by spatial transcriptomics [visium]/S07/GSM6215671_S07.png',
        '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Zika virus co-opts miRNA networks to persist in placental microenvironments detected by spatial transcriptomics [visium]/S07/GSM6215671_S07_filtered_feature_bc_matrix.h5',
        '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Zika virus co-opts miRNA networks to persist in placental microenvironments detected by spatial transcriptomics [visium]/S07/alignment.json')
    
   # meta_df = pd.read_csv('/mnt/ssd/paul/ST-histology-loader/data/Datasets H&E spatial transcriptomics - 10XGenomics.csv')
    #meta_df = pd.read_csv('Datasets H&E spatial transcriptomics - Zenodo.csv')
    #download_from_meta_df(meta_df,
    #                      '/mnt/ssd/paul/ST-histology-loader/data/samples')
    
    #process_all(meta_df[51:], sample_path)

    # adata_shape_l_unprocessed = plot_unprocessed(st_dir, img_dir, df_img_name, f'./figures/{project}/unprocessed_plots')
    save_path_df_metrics = f'./figures/{project}/df_metrics_{project}.csv'
    load_img = True
    if project == 'crc_14':
        load_img = False
        
    

    
    #adata_list, img_list, hvgs_union, sample_names, matched_dict = load_data(st_dir, img_dir, df_img_name)
    adata_list, img_list, sample_names = load_data(data_path)
    
    start, end = 0, len(adata_list)
    # # Ensure the directory exists for saving the figure
    dist_img_save_dir = f'./figures/{project}/combined_dist_plots'
    spatial_img_save_dir = f'./figures/{project}/spatial_plots'

    #plot_dist_plots(adata_list, sample_names, dist_img_save_dir, start, end)
    plot_spatial_plots(adata_list, sample_names, spatial_img_save_dir, start, end)
    
    for i in range(len(adata_list)):
        patches = extract_image_patches(adata_list[i], img_list[i], patch_size = 200)
    #visualize_patches(patches[:20])
    mask_bad = identify_bad_patches(patches[:200])
    visualize_patches([patches[i] for i in mask_bad])

if __name__ == "__main__":
    main()