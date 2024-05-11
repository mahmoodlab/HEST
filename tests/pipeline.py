import sys

import pandas as pd

sys.path.append("/mnt/ssd/paul/ST-histology-loader")

import cProfile
from cProfile import Profile
from pstats import SortKey, Stats

import numpy as np
import pyvips
import scanpy as sc
import tifffile
from packaging import version
from PIL import Image

from src.hest.HESTData import create_benchmark_data, create_splits
from src.hest.readers import process_meta_df
from src.hest.utils import copy_processed_images, get_k_genes_from_df


def create_gene_panels(name_to_meta):
    
    for key, val in name_to_meta.items():
        get_k_genes_from_df(val, k=50, save_dir=f"/mnt/sdb1/paul/{key}/var_50genes.json", criteria='var')


def main():

    exclude_list = [
        'Spatial Transcriptomic Experiment of Triple-Negative Breast Cancer PDX Model PIM001-P model treatment naive sample',
        'Visium Spatial Gene Expression of embryonic mouse brain at embryonic day 15.5',
        'Dissecting the melanoma ecosystem one cell at the time during immunotherapy',
        'Spatial Transcriptomics of human fetal liver',
        'Spatiotemporal mapping of immune and stem cell dysregulation after volumetric muscle loss',
        'Spatial transcriptomics profiling of the developing mouse embryo'
    ]

    #meta = '/media/ssd2/hest/HEST_v0_0_1.csv'
    #meta = '/media/ssd2/hest/ST H&E datasets - Combined data.csv'
    meta = '/mnt/sdb1/paul/data/samples/ST H&E datasets - Combined data.csv'
    #meta = '/mnt/sdb1/paul/meta_releases/HEST_v0_0_1.csv'
    meta_df = pd.read_csv(meta)

    if meta.endswith('HEST_v0_0_1.csv'):
        meta_df['image'] = [True for _ in range(len(meta_df))]
    #meta_df = meta_df[meta_df['Products'] == 'Spatial Gene Expression']
    meta_df = meta_df[meta_df['image'] == True]
    meta_df = meta_df[meta_df['st_technology'] != 'Visium HD']
    #meta_df = meta_df[meta_df['st_instrument'] != 'Xenium Analyzer']
    meta_df = meta_df[~meta_df['dataset_title'].isin(exclude_list)]
    #meta_df = meta_df[meta_df['bigtiff'].notna() & meta_df['bigtiff'] != "TRUE"]
    #meta_df = meta_df[(meta_df['id'] == "TENX138")]
    #meta_df = meta_df[meta_df['id'].str.startswith('GIT')]
    #meta_df = meta_df[(meta_df['id'] == "NCBI783") | (meta_df['id'] == "NCBI785")]
    #meta_df = meta_df[(meta_df['id'] == "SPA100")]

    #meta_df = meta_df[(meta_df['use_train'] == 'TRUE')]
    #img = pyvips.Image.tiffload('/mnt/sdb1/paul/images/pyramidal/NCBI844.tif')

    dest = '/mnt/sdb1/paul/images'
    
    #df = pool_xenium_by_cell('/mnt/sdb1/paul/data/samples/xenium/FFPE Human Breast using the Entire Sample Area/Tissue sample 1', '/mnt/sdb1/paul/TENX95_cell_detection.geojson', 
    #                         0.2125)
    #df.to_parquet('TENX95_pool.parquet')

    #df = pool_xenium_by_cell('/mnt/sdb1/paul/data/samples/xenium/FFPE Human Breast using the Entire Sample Area/Replicate 1', '/mnt/sdb1/paul/TENX99_cell_detection.geojson', 
    #                         0.2125)
    #df.to_parquet('TENX99_pool.parquet')
    
    #df = pool_xenium_by_cell('/mnt/sdb1/paul/data/samples/xenium/High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis [Xenium]/Breast Cancer, Xenium In Situ Spatial Gene Expression Rep 1/', '/mnt/sdb1/paul/NCBI785_cell_detection.geojson', 
    #                         0.2125)#0.3639107956749145)
    #df.to_parquet('NCBI785_pool.parquet')
    
    #df = pool_xenium_by_cell('/mnt/sdb1/paul/data/samples/xenium/High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis [Xenium]/Breast Cancer, Xenium In Situ Spatial Gene Expression', '/mnt/sdb1/paul/NCBI783_cell_detection.geojson', 
    #                         0.27395985597740874)
    #df.to_parquet('NCBI783_pool.parquet')
    
    #meta_df = meta_df[(meta_df['dataset_title'] == "Multimodal decoding of human liver regeneration [st_human]") | (meta_df['dataset_title'] == "Multimodal decoding of human liver regeneration [st_mouse]")] 
    #root = '/mnt/sdb1/paul/data/samples/visium/Charting the Heterogeneity of Colorectal Cancer Consensus Molecular Subtypes using Spatial Transcriptomics: datasets/A595688_Rep1/processed/'
    ##read_HESTData(root + '/aligned_adata.h5ad', 
    #             '/mnt/sdb1/paul/images/pyramidal/ZEN42.tif', 
    #              root + '/metrics.json')
    
    
    name_to_meta = {
        'IDC_ILC': meta_df[(meta_df['id'] == "TENX99") | (meta_df['id'] == "TENX95") | (meta_df['id'] == "NCBI785") | (meta_df['id'] == "NCBI783") | (meta_df['id'] == "TENX94")],
        #'BC2': meta_df[(meta_df['dataset_title'] == "Integrating spatial gene expression and breast tumour morphology via deep learning")], #BC2
        'SCC': meta_df[(meta_df['dataset_title'] == "Single Cell and Spatial Analysis of Human Squamous Cell Carcinoma [ST]")], #SCC
        #'BC1': meta_df[(meta_df['dataset_title'] == "Spatial deconvolution of HER2-positive breast cancer delineates tumor-associated cell type interactions")], #BC1
        'PAAD': meta_df[(meta_df['id'] == "TENX126") | (meta_df['id'] == "TENX116")],
        'FHPTLD': meta_df[(meta_df['id'] == "TENX124") | (meta_df['id'] == "TENX125")],
        'SKCM': meta_df[(meta_df['id'] == "TENX117") | (meta_df['id'] == "TENX115")],
        'CRC_COAD': meta_df[(meta_df['oncotree_code'] == 'COAD') & (meta_df['dataset_title'] == "Charting the Heterogeneity of Colorectal Cancer Consensus Molecular Subtypes using Spatial Transcriptomics: datasets")],
        'CRC_READ':meta_df[(meta_df['oncotree_code'] == 'READ') & (meta_df['dataset_title'] == "Charting the Heterogeneity of Colorectal Cancer Consensus Molecular Subtypes using Spatial Transcriptomics: datasets")],
        'IDC': meta_df[(meta_df['id'] == "TENX99") | (meta_df['id'] == "TENX95") | (meta_df['id'] == "NCBI785") | (meta_df['id'] == "NCBI783")],
        'PRAD': meta_df[meta_df['dataset_title'] == 'Spatially resolved clonal copy number alterations in benign and malignant tissueJus'],
        'LYMPH_IDC': meta_df[meta_df['dataset_title'] == 'Single cell profiling of primary and paired metastatic lymph node tumors in breast cancer patients'],
        'CCRCC': meta_df[meta_df['dataset_title'] == 'Tertiary lymphoid structures generate and propagate anti-tumor antibody-producing plasma cells in renal cell cancer'],
        'HCC': meta_df[meta_df['dataset_title'] == 'Identification of TREM1+CD163+ myeloid cells as a deleterious immune subset in HCC [Spatial Transcriptomics]']
    }
    
    meta_df = name_to_meta['CCRCC']  #meta_df[meta_df['id'] == "TENX94"] #name_to_meta['IDC_ILC']
    #name_to_meta
    
    #create_gene_panels(name_to_meta)
    
    #bc1_splits = meta_df.groupby('patient')['id'].agg(list).to_dict()
    #create_splits('/mnt/sdb1/paul/bc2_splits', bc1_splits, K=8)
    #test = sc.read_h5ad('/mnt/sdb1/paul/BC1/adata/SPA154.h5ad')
    
    
    #process_meta_df(meta_df, pyramidal=False)
    #get_k_genes_from_df(meta_df, k=50, save_dir="/mnt/sdb1/paul/SKCM/var_50genes.json", criteria='var')
    
    #copy_processed_images(dest, meta_df, cp_pyramidal=False)
    
    #pyvips.tiffload('/media/ssd2/hest/pyramidal/TENX137.tif'
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/CCRCC', K=6, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/IDC_ILC', K=None, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/CCRCC', K=12, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/SCC', K=4, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/HCC', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/LYMPH_IDC', K=4, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/PRAD', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/CRC_READ', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/CRC_COAD', K=3, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/SKCM', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/FHPTLD', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/PAAD', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/BC1', K=8, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/BC2', K=8, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/SCC', K=4, adata_folder='/mnt/sdb1/paul/images/adata')
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/IDC', K=4, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/CRC', K=7, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/PROST1', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #mask_and_patchify(meta_df, '/mnt/sdb1/paul/patches', use_mask=False)

    
    #create_meta_release(meta_df, version.Version('0.0.2'))

if __name__ == "__main__":
    main()