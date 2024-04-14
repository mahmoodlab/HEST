#import apeer_ometiff_library.io
import pandas as pd
from src.hiSTloader.old_st import *
from src.hiSTloader.helpers import save_spatial_plot, GSE184384_to_h5, write_10X_h5

import tifffile
from PIL import Image
from kwimage.im_cv2 import warp_affine, imresize
import subprocess
#import fiftyone as fo
from tqdm import tqdm

import shutil
import json
#from valis import registration

Image.MAX_IMAGE_PIXELS = 9331200000
    
        
        
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
        
    tech = row['st_instrument']
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
 
    
def process_meta_df(meta_df, save_spatial_plots=True):
    for index, row in tqdm(meta_df.iterrows()):
        path = _get_path_from_meta_row(row)
        adata = read_any(path)
        if save_spatial_plots:
            save_spatial_plot(adata, os.path.join(path, 'processed'), 'test')
        

def copy_processed_images(dest, meta_df, cp_spatial=True, cp_downscaled=True):
    for index, row in meta_df.iterrows():
        
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
            path_dest_downscaled = os.path.join(dest, 'downscaled', row['id'] + '_downscaled_fullres.jpeg')
            shutil.copy(path_downscaled, path_dest_downscaled)
        if cp_spatial:
            path_spatial = os.path.join(path, 'spatial_plots.png')
            path_dest_spatial = os.path.join(dest, 'spatial_plots', row['id'] + '_spatial_plots.png')
            shutil.copy(path_spatial, path_dest_spatial)
        path_dest_fullres = os.path.join(dest, 'fullres', row['id'] + '_aligned_fullres_HE.ome.tif')
        

        
        #path_downscaled = os.path.join(path, 'downscaled_fullres.jpeg')
        #path_dest_downscaled = os.path.join(dest, 'downscaled', row['id'] + '_downscaled_fullres.jpeg')
        
        shutil.copy(path_fullres, path_dest_fullres)
        

def open_fiftyone():
    dest = '/mnt/sdb1/paul/images'
    dataset = fo.Dataset.from_images_dir("/mnt/sdb1/paul/images")
    session = fo.launch_app(dataset)    
    
    
def create_joined_gene_plots(meta):
    # determine common genes
    common_genes = None
    n = len(meta)
    for index, row in meta.iterrows():
        path = _get_path_from_meta_row(row)
        gene_files = np.array(os.listdir(os.path.join(path, 'gene_bar_plots')))
        if common_genes is None:
            common_genes = gene_files
        else:
            common_genes = np.intersect1d(common_genes, gene_files)
            
    for gene in tqdm(common_genes):
        fig, axes = plt.subplots(n, 1)
        i = 0
        for index, row in meta.iterrows():
            path = _get_path_from_meta_row(row)
            gene_path = os.path.join(path, 'gene_bar_plots', gene)
            image = Image.open(gene_path)
            axes[i].imshow(image)
            axes[i].axis('off')
            i += 1
        plt.savefig(os.path.join('/mnt/sdb1/paul/gene_subplots', f'{gene}_subplot.png'), bbox_inches='tight', pad_inches=0, dpi=600)
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
    
    


def main():

    path = '/mnt/sdb1/paul/data/samples/visium/Visium CytAssist Gene Expression Libraries of Post-Xenium Human Colon Cancer (FFPE)/Control, Replicate 1/processed'
    
    meta = '/mnt/sdb1/paul/data/samples/ST H&E datasets - Combined data.csv'
    meta_df = pd.read_csv(meta)
    #meta_df = meta_df[meta_df['Products'] == 'Spatial Gene Expression']
    #meta_df = meta_df[meta_df['id'].str.startswith('MEND')]
    
    #GSE184384_to_h5('/mnt/sdb1/paul/data/samples/visium/Epithelial Plasticity and Innate Immune Activation Promote Lung Tissue Remodeling following Respiratory Viral Infection./R3_Spatial')

    
    
    exclude_list = [
        'Spatial Transcriptomic Experiment of Triple-Negative Breast Cancer PDX Model PIM001-P model treatment naive sample',
        'Spatial detection of fetal marker genes expressed at low level in adult human heart tissue',
        'Tertiary lymphoid structures generate and propagate anti-tumor antibody-producing plasma cells in renal cell cancer',
        'Bern ST',
        'Spatial architecture of high-grade glioma reveals tumor heterogeneity within distinct domains'
    ]
    
    meta_df = meta_df[meta_df['image'] == True]
    meta_df = meta_df[meta_df['Products'] != 'HD Spatial Gene Expression']
    meta_df = meta_df[~meta_df['dataset_title'].isin(exclude_list)]
    #meta_df = meta_df[meta_df['dataset_title'] == 'Spatially resolved clonal copy number alterations in benign and malignant tissueJus']
    #meta_df = meta_df[meta_df['dataset_title'] == 'Spatially resolved clonal copy number alterations in benign and malignant tissueJus']
    #meta_df = meta_df[meta_df['dataset_title'] == 'FFPE Human Breast using the Entire Sample Area']
    #meta_df = meta_df[meta_df['check_image'] == "TRUE"]
    
    #meta_df = meta_df[((meta_df['dataset_title'] == 'FFPE Human Breast using the Entire Sample Area') & (meta_df['subseries'] == 'Replicate 1')) |
    #              ((meta_df['dataset_title'] == 'FFPE Human Breast with Pre-designed Panel') & (meta_df['subseries'] == 'Tissue sample 1')) |
    #              ((meta_df['dataset_title'] == 'High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis [Xenium]') & (meta_df['subseries'] != 'Breast Cancer, Xenium In Situ Spatial Gene Expression Rep 2'))]
    #

    #create_joined_gene_plots(meta_df)

    dest = '/mnt/sdb1/paul/images'
    #copy_processed_images(dest, meta_df, cp_spatial=False, cp_downscaled=False)

    #open_fiftyone()
    #copy_processed
    # images(dest, meta_df)
    
    process_meta_df(meta_df[299:], save_spatial_plots=True) #230

    prefix = "/mnt/sdb1/paul/data/samples/visium/Tertiary lymphoid structures generate and propagate anti-tumor antibody-producing plasma cells in renal cell cancer"
    #prefix = "/mnt/sdb1/paul/data/samples/visium/Prostate ST Internal"
    
    #GSE184369_split_to_h5ad(prefix)
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
        

if __name__ == "__main__":
    main()