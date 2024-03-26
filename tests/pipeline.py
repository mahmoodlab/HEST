import pandas as pd
from src.hiSTloader.old_st import *
from src.hiSTloader.helpers import join_object_to_adatas_GSE171351, process_all, extract_patch_expression_pairs, visualize_patches, extract_image_patches, download_from_meta_df, read_10x_visium, save_10x_visium, save_spatial_plot \
    , join_object_to_adatas_GSE214989

import tifffile
from PIL import Image
from kwimage.im_cv2 import warp_affine, imresize
import openslide

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
    

def cut_in_four_ndpi(path):
    img = openslide.OpenSlide(path)
    height, width = img.dimensions
    img1 = img.read_region((0, 0), 0, (width // 4, height)) #img[:, :width // 4]
    myimg = imresize(img1, 0.025)
    plt.imshow(myimg)
    plt.show()
    img2 = img.read_region((width // 4, 0), 0, (width // 4, height)) #img[:, width // 4:(2 * width) // 4]
    img3 = img.read_region((width // 2, 0), 0, (width // 4, height)) #img[:,(2 * width) // 4:(3 * width) // 4]
    img4 = img.read_region(((3 * width) // 4, 0), 0, (width // 4, height)) #img[:, (3 * width) // 4:]
    
    with tifffile.TiffWriter('split1.ome.tif', bigtiff=True) as tif:
        tif.write(img1)
    with tifffile.TiffWriter('split2.ome.tif', bigtiff=True) as tif:
        tif.write(img2)
    with tifffile.TiffWriter('split3.ome.tif', bigtiff=True) as tif:
        tif.write(img3)
    with tifffile.TiffWriter('split4.ome.tif', bigtiff=True) as tif:
        tif.write(img4)


def cut_in_four_from_bbox(path, bbox):
    left_corner = bbox[0]
    
    img = openslide.OpenSlide(path)
    height, width = img.dimensions
    img1 = img.read_region((0, 0), 0, (width // 4, height)) #img[:, :width // 4]
    img2 = img.read_region((width // 4, 0), 0, (width // 4, height)) #img[:, width // 4:(2 * width) // 4]
    img3 = img.read_region((width // 2, 0), 0, (width // 4, height)) #img[:,(2 * width) // 4:(3 * width) // 4]
    img4 = img.read_region(((3 * width) // 4, 0), 0, (width // 4, height)) #img[:, (3 * width) // 4:]
    
    with tifffile.TiffWriter('split1.ome.tif', bigtiff=True) as tif:
        tif.write(img1)
    with tifffile.TiffWriter('split2.ome.tif', bigtiff=True) as tif:
        tif.write(img2)
    with tifffile.TiffWriter('split3.ome.tif', bigtiff=True) as tif:
        tif.write(img3)
    with tifffile.TiffWriter('split4.ome.tif', bigtiff=True) as tif:
        tif.write(img4)



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
    
    
def _create_dir_structure_from_meta_df(meta_df):
    for index, row in meta_df.iterrows():
        dataset_title = row['dataset_title']
        subseries = row['subseries']
        path = os.path.join('/mnt/sdb1/paul/data/samples/visium', dataset_title, subseries)
        os.makedirs(path, exist_ok=True)
        
        

def main():
    project = 'test_paul'
    sample_path = '/mnt/ssd/paul/ST-histology-loader/data/samples'
    
    #meta_df = pd.read_csv('/mnt/sdb1/paul/ST H&E datasets - 10XGenomics.csv')[34:]
    #process_all(meta_df, '/mnt/sdb1/paul/data/samples')
    
    
    #_create_dir_structure_from_meta_df(meta_df[80:])
    
    #adata = sc.read_h5ad('/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Spatial transcriptomics profiling of the developing mouse embryo/Embryo, E12.5,  visium/_1_filtered_feature_bc_matrix.h5')
    
    
    
    
    #path = '/mnt/sdb1/paul/data/samples/visium/Bern ST/2000233V11Y17-0362022-05-17 - 2022-03-07 12.42.15.ndpi'
    #cut_in_four_ndpi(path)
    
   # adata = sc.read_10x_h5('/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Distinct mesenchymal cell states mediate prostate cancer progression [Spatial Transcriptomics]/PRN/GSM7914975_WT_feature_bc_matrix.h5')
    
    #read_any()
    
   # my = sc.read_visium('/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Mouse Brain Serial Section 1 (Sagittal-Posterior)')
    

    #path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Spatial transcriptomics profiling of the developing mouse embryo/GSE214989_counts_embryo_visium.h5'
    #join_object_to_adatas_GSE214989(path)
     
    
    #path = '/mnt/sdb1/paul/data/samples/visium/A new epithelial cell subpopulation predicts response to surgery, chemotherapy, and immunotherapy in bladder cancer/GSE171351_combined_visium.h5ad'
    #join_object_to_adatas_GSE171351(path)
    
    
    #path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/YAP Drives Assembly of a Spatially Colocalized Cellular Triad Required for Heart Renewal/MCM control mouse heart spatial RNA-seq'
   # path = "/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Multi-resolution deconvolution of spatial transcriptomics data reveals continuous patterns of inflammation"
    #path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Spatial transcriptomics of adenoid cystic carcinoma of the lacrimal gland/LGACC, sample A, spatial'
    #path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/10X Visium Spatial transcriptomics of murine colon in steady state and during recovery after DSS colitis/Colon DSS day0'
    #path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Epithelial Plasticity and Innate Immune Activation Promote Lung Tissue Remodeling following Respiratory Viral Infection./R3_Spatial'
    path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Spatially resolved transcriptomics reveals the architecture of the tumor-microenvironment interface/Visium-A'
    #path = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Assessment of Spatial Genomics for Oncology Discovery'
    #path = '/mnt/sdb1/paul/data/samples/visium/Spatial transcriptomics of the mouse brain across three age groups/Middle Replicate 2 Slide 2'
    
    #read_any(path)
    #path = '/mnt/sdb1/paul/data/samples/visium/Single-cell and spatial transcriptomics characterisation of the immunological landscape in the healthy and PSC human liver'
    #
    # 
    # dirs = ['Tumor_1', 'Tumor_2', 'LN_1', 'LN_2']


    #imagepath = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/10X Visium Spatial transcriptomics of murine colon in steady state and during recovery after DSS colitis/Colon DSS day0/GSM5213483_V19S23-097_A1_S1_Region1_colon_d0_rotated.tif'
    
    #img = np.array(Image.open(imagepath))
    #img = tifffile.imread(imagepath)
    #img2 = imresize(img, 0.025)
    #plt.imshow(img2)
    #plt.show()
    
    #for mypath in tqdm(os.listdir(path)[12:]):
    #mypath = os.path.join(path, mypath)
    #paths = [
        #'/mnt/sdb1/paul/data/samples/visium/Spatial transcriptomics of the mouse brain across three age groups/Old Replicate 1 Slide 1',
        #'/mnt/sdb1/paul/data/samples/visium/Spatial transcriptomics of the mouse brain across three age groups/Old Replicate 2 Slide 2',
    #    '/mnt/sdb1/paul/data/samples/visium/Spatial transcriptomics of the mouse brain across three age groups/Young Replicate 1 Slide 1',
     #   '/mnt/sdb1/paul/data/samples/visium/Spatial transcriptomics of the mouse brain across three age groups/Young Replicate 2 Slide 1'
    #]
    #prefix = '/mnt/sdb1/paul/data/samples/visium/Single cell profiling of primary and paired metastatic lymph node tumors in breast cancer patients'
    #prefix = '/mnt/sdb1/paul/data/samples/visium/Single-cell and spatial transcriptomics characterisation of the immunological landscape in the healthy and PSC human liver'
    #prefix = '/mnt/sdb1/paul/data/samples/visium/A cellular hierarchy in melanoma uncouples growth and metastasis'
    #prefix = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Spatial RNA sequencing of regenerating mouse hindlimb muscle'
    
    #path = '/mnt/sdb1/paul/data/samples/visium/Spatial sequencing of Foreign body granuloma/CMC_Visium_Sponge_2 wks'

    
    #prefix = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Spatial transcriptomics profiling of the developing mouse embryo'
    #prefix = '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/10X Visium Spatial transcriptomics of murine colon in steady state and during recovery after DSS colitis'
    
    prefix ='/mnt/sdb1/paul/data/samples/visium/Single-nucleus Ribonucleic Acid-sequencing and Spatial Transcriptomics Reveal the Cardioprotection of Shexiang Baoxin Pill (MUSKARDIA) in Mice with Myocardial Ischemia-Reperfusion Injury'
    paths = os.listdir(prefix)
    for path in paths:
        path = os.path.join(prefix, path)
        adata, spatial_aligned, img, raw_bc_matrix = read_any(path)
        save_spatial_plot(adata, os.path.join(path, 'processed'), 'test')
        
    

    
    for dir in dirs:
        mypath = os.path.join(path, dir)
    
        adata, spatial_aligned, img, raw_bc_matrix = read_any(mypath)
    
        save_spatial_plot(adata, os.path.join(mypath, 'processed'), 'test')
    
    
    #adata, spatial_aligned, img, raw_bc_matrix = read_10x_visium(
    #    img_path='/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Mouse Brain Serial Section 1 (Sagittal-Posterior)/V1_Mouse_Brain_Sagittal_Posterior_image.tif',
    #    bc_matrix_path='/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Mouse Brain Serial Section 1 (Sagittal-Posterior)/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5',
    #    spatial_coord_path='/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Mouse Brain Serial Section 1 (Sagittal-Posterior)/spatial'
    #)
    
    # provide an option to keep the spatial folder with/without the tissue_positions.csv
    #save_10x_visium(
    #    adata, 
    #    '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Mouse Brain Serial Section 1 (Sagittal-Posterior)/processed',
    #    img,
    #    h5_path='/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Mouse Brain Serial Section 1 (Sagittal-Posterior)/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5',
    #    spatial_path='/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Mouse Brain Serial Section 1 (Sagittal-Posterior)/spatial',
    #)
    
    #load_from_image_matrix_align(
    #    '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Zika virus co-opts miRNA networks to persist in placental microenvironments detected by spatial transcriptomics [visium]/S07/GSM6215671_S07.png',
    #    '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Zika virus co-opts miRNA networks to persist in placental microenvironments detected by spatial transcriptomics [visium]/S07/GSM6215671_S07_filtered_feature_bc_matrix.h5',
    #    '/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Zika virus co-opts miRNA networks to persist in placental microenvironments detected by spatial transcriptomics [visium]/S07/alignment.json')
    
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