import pandas as pd
from src.hiSTloader.old_st import *
from src.hiSTloader.helpers import join_object_to_adatas_GSE171351, process_all, extract_patch_expression_pairs, visualize_patches, extract_image_patches, download_from_meta_df, read_10x_visium, save_10x_visium, save_spatial_plot \
    , join_object_to_adatas_GSE214989, GSE234047_to_h5, GSE206391_split_h5, _GSE206391_copy_dir, GSE184369_to_h5, read_ST, save_spatial_metrics_plot, filter_adata, save_metrics_plot, GSE236787_split_to_h5
from src.hiSTloader.align import autoalign_with_fiducials
import spatialdata_io
import spatialdata_plot

import tifffile
from PIL import Image
from kwimage.im_cv2 import warp_affine, imresize
import subprocess

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
    #myimg = imresize(img1, 0.025)
    #plt.imshow(myimg)
    #plt.show()
    img2 = img.read_region((width // 4, 0), 0, (width // 4, height)) #img[:, width // 4:(2 * width) // 4]
    img3 = img.read_region((width // 2, 0), 0, (width // 4, height)) #img[:,(2 * width) // 4:(3 * width) // 4]
    img4 = img.read_region(((3 * width) // 4, 0), 0, (width // 4, height)) #img[:, (3 * width) // 4:]
    
    base_dir = os.path.dirname(path)
    name = os.path.basename(path).split('.')[0]
    
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split1.ome.tif'), bigtiff=True) as tif:
        tif.write(img1)
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split2.ome.tif'), bigtiff=True) as tif:
        tif.write(img2)
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split3.ome.tif'), bigtiff=True) as tif:
        tif.write(img3)
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split4.ome.tif'), bigtiff=True) as tif:
        tif.write(img4)


def cut_in_four_from_bbox(path):
    
    #img = openslide.OpenSlide(path)
    img = tifffile.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    img1 = img[:,:width // 4]
    img2 = img[:,width // 4:(2 * width) // 4]
    img3 = img[:,(2 * width) // 4:(3 * width) // 4]
    img4 = img[:,(3 * width) // 4:]
    #img1 = img.read_region((0, 0), 0, (width // 4, height)) #img[:, :width // 4]
    #img2 = img.read_region((width // 4, 0), 0, (width // 4, height)) #img[:, width // 4:(2 * width) // 4]
    #img3 = img.read_region((width // 2, 0), 0, (width // 4, height)) #img[:,(2 * width) // 4:(3 * width) // 4]
    #img4 = img.read_region(((3 * width) // 4, 0), 0, (width // 4, height)) #img[:, (3 * width) // 4:]
    
    base_dir = os.path.dirname(path)
    name = os.path.basename(path).split('.')[0]
    
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split1.ome.tif'), bigtiff=True) as tif:
        tif.write(img1, compression='zlib', compressionargs={'level': 8})
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split2.ome.tif'), bigtiff=True) as tif:
        tif.write(img2, compression='zlib', compressionargs={'level': 8})
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split3.ome.tif'), bigtiff=True) as tif:
        tif.write(img3, compression='zlib', compressionargs={'level': 8})
    with tifffile.TiffWriter(os.path.join(base_dir, f'{name}_split4.ome.tif'), bigtiff=True) as tif:
        tif.write(img4, compression='zlib', compressionargs={'level': 8})



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
        if isinstance(row['subseries'], float):
            continue
        dataset_title = row['dataset_title']
        subseries = row['subseries']
        path = os.path.join('/mnt/sdb1/paul/data/samples/visium', dataset_title, subseries)
        os.makedirs(path, exist_ok=True)
        
        
def read_bern():
    path = '/mnt/sdb1/paul/data/samples/visium/Bern ST'
    df = pd.read_csv(os.path.join(path, 'mouse_mammary_tumor_meta_df.csv'))
    i = 0
    for index, row in tqdm(df.iterrows()):
        bbox = eval(row['bounding_box'])
        px_size = 0.2299
        x = int(bbox[0][0] / px_size)
        y = int(bbox[0][1] / px_size)
        w = int((bbox[2][0] - bbox[0][0]) / px_size)
        h = int((bbox[2][1] - bbox[0][1]) / px_size)
        img_path = os.path.join(path, row['slide_name'])
        
        x -= 1500
        y -= 1000
        
        img=tifffile.imread(img_path, series=0, level=0)
        img2 = img[y:y+h, x:x+w]
        with tifffile.TiffWriter(os.path.join(path, f'my{i}_split1.ome.tif'), bigtiff=True) as tif:
            tif.write(img2, compression='zlib', compressionargs={'level': 8})
        
        #subprocess.Popen(f'"/mnt/sdb1/paul/data/samples/visium/Bern ST/ndpisplit" -Ex40,{x},{y},{w},{h},i{i} "{img_path}"', shell=True)
        #subprocess.call(['/mnt/sdb1/paul/data/samples/visium/Bern ST/ndpisplit', f'-Ex40,{x},{y},{w},{h},i{i}', f'{img_path}'])
        i += 1
        #break
    
    print(bbox)
    
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
    
    
def process_meta_df(meta_df):
    for index, row in meta_df.iterrows():
        subseries = row['subseries']
        if isinstance(subseries, float):
            subseries = ""
        path = os.path.join('/mnt/sdb1/paul/data/samples/visium', row['dataset_title'], subseries)
        adata = read_any(path)
        save_spatial_plot(adata, os.path.join(path, 'processed'), 'test')
    

def main():
    
    path = '/mnt/sdb1/paul/data/samples/visium/Visium CytAssist Gene Expression Libraries of Post-Xenium Human Colon Cancer (FFPE)/Control, Replicate 1/processed'
    #vs = spatialdata_io.visium(path, dataset_id='', fullres_image_file='aligned_fullres_HE.ome.tif')
    
    #vs.pl.render_images().pl.render_shapes().pl.show("global")
    #GSE236787_split_to_h5(path)
    
    meta = '/mnt/sdb1/paul/ST H&E datasets - 10XGenomics.csv'
    meta_df = pd.read_csv(meta)[32:]
    meta_df = meta_df[meta_df['Products'] == 'Spatial Gene Expression']
    meta_df = meta_df[meta_df['image'] == True]
    
    process_meta_df(meta_df)

    prefix = "/mnt/sdb1/paul/data/samples/visium/A novel model of binge ethanol exposure reveals enhanced neurodegeneration with advanced age"
    
    
    #adata = read_any(prefix)
    #filtered_adata = filter_adata(adata)
    #save_spatial_plot(adata, os.path.join(prefix, 'processed'), 'test')
    #save_metrics_plot(adata, os.path.join(prefix, 'processed'), 'test')
    #save_spatial_metrics_plot(adata, os.path.join(prefix, 'processed'), 'test', filtered_adata=filtered_adata)
    paths = os.listdir(prefix)
    # Use list comprehension to filter out non-directory items
    folders_only = [item for item in paths if os.path.isdir(os.path.join(prefix, item))]
    if len(folders_only) != len(paths):
        paths = [prefix]
    exclude_paths = []
    for path in paths:
        if path == 'old' or path in exclude_paths:
            continue
        path = os.path.join(prefix, path)

        adata = read_any(path)
        save_spatial_plot(adata, os.path.join(path, 'processed'), 'test')
        #save_metrics_plot(adata, os.path.join(path, 'processed'), 'test')
        #save_metrics_plot(adata, os.path.join(path, 'processed'), 'test')
        #break
        

if __name__ == "__main__":
    main()