import sys

import pandas as pd

from src.hest.bench.training.predict_expression import benchmark_encoder

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
from src.hest.readers import VisiumReader, process_meta_df, VisiumHDReader
from src.hest.utils import copy_processed_images, get_k_genes_from_df
import pyreadr



def create_gene_panels(meta_df):
    
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
    #meta_df = meta_df[meta_df['dataset_title'] == 'Batf3-dendritic cells and 4-1BB-4-1BB ligand axis are required at the effector phase within the tumor microenvironment for PD-1-PD-L1 blockade efficacy']
    #meta_df = meta_df[meta_df['dataset_title'] == 'A new epithelial cell subpopulation predicts response to surgery, chemotherapy, and immunotherapy in bladder cancer']
    #meta_df = meta_df[(meta_df['dataset_title'] == 'Preview Data: FFPE Human Lung Cancer with Xenium Multimodal Cell Segmentation') | (meta_df['dataset_title'] == 'FFPE Human Lung Cancer Data with Human Immuno-Oncology Profiling Panel and Custom Add-on')]
    
    name_to_meta = {
        'IDC_ILC': meta_df[(meta_df['id'] == "TENX99") | (meta_df['id'] == "TENX95") | (meta_df['id'] == "NCBI785") | (meta_df['id'] == "NCBI783") | (meta_df['id'] == "TENX94")],
        #'BC2': meta_df[(meta_df['dataset_title'] == "Integrating spatial gene expression and breast tumour morphology via deep learning")], #BC2
        'SCC': meta_df[(meta_df['dataset_title'] == "Single Cell and Spatial Analysis of Human Squamous Cell Carcinoma [ST]")], #SCC
        #'BC1': meta_df[(meta_df['dataset_title'] == "Spatial deconvolution of HER2-positive breast cancer delineates tumor-associated cell type interactions")], #BC1
        'PAAD': meta_df[(meta_df['id'] == "TENX126") | (meta_df['id'] == "TENX116") | (meta_df['id'] == "TENX140")],
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
    
    #meta_df = name_to_meta['PAAD']
    #meta_df = meta_df[(meta_df['id'] == 'TENX139') | (meta_df['id'] == 'TENX142')]
    meta_df = meta_df[meta_df['st_technology'] != 'Xenium']

    #meta_df = meta_df[(meta_df['use_train'] == 'TRUE')]
    #img = pyvips.Image.tiffload('/mnt/sdb1/paul/images/pyramidal/NCBI844.tif')

    dest = '/mnt/sdb1/paul/images'
    
    #img = pyvips.Image.svgload('/mnt/sdb1/paul/Nor_lung.svg', unlimited=True)
    
    #from src.hest.bench import benchmark_encoder
    #import torch.nn as nn
    
    
    #benchmark_encoder(
    #    None, 
    #    None,
    #    '/mnt/ssd/paul/ST-histology-loader/samples/bench_config.yaml'
    #)
    
    #obj = VisiumReader().auto_read('/mnt/sdb1/paul/data/samples/visium/Batf3-dendritic cells and 4-1BB-4-1BB ligand axis are required at the effector phase within the tumor microenvironment for PD-1-PD-L1 blockade efficacy/GSM7659430')
    #obj.save_spatial_plot('/mnt/sdb1/paul/data/samples/visium/Batf3-dendritic cells and 4-1BB-4-1BB ligand axis are required at the effector phase within the tumor microenvironment for PD-1-PD-L1 blockade efficacy/GSM7659430/processed')

    process_meta_df(meta_df, pyramidal=False)
    #get_k_genes_from_df(meta_df, k=50, save_dir="/mnt/ssd/paul/ST-histology-loader/bench_data/PAAD/var_50genes.json", criteria='var')
    
    #copy_processed_images(dest, meta_df, cp_pyramidal=False)
    
    #pyvips.tiffload('/media/ssd2/hest/pyramidal/TENX137.tif'
    #create_benchmark_data(meta_df, save_dir='/mnt/ssd/paul/ST-histology-loader/bench_data/PAAD', K=3, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True, keep_largest=[False, True, True])
    #create_benchmark_data(meta_df, save_dir='/mnt/ssd/paul/ST-histology-loader/bench_data/LUNG', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=True, keep_largest=[True, False])
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/BLAD2', K=4, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/EPM', K=11, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
    #create_benchmark_data(meta_df, save_dir='/mnt/sdb1/paul/BLAD', K=2, adata_folder='/mnt/sdb1/paul/images/adata', use_mask=False)
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