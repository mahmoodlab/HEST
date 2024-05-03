import pandas as pd

import sys
sys.path.append("/media/ssd2/hest/hest")

from src.hest.utils import create_meta_release, copy_processed_images, create_joined_gene_plots, get_k_mean_expressed_genes_from_df
from src.hest.patching import mask_spots, mask_and_patchify, create_splits
from src.hest.readers import read_and_save, process_meta_df
from packaging import version
from PIL import Image
import tifffile
import numpy as np
import scanpy as sc
import cProfile
from pstats import SortKey, Stats
from cProfile import Profile

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
    meta = '/media/ssd2/hest/ST H&E datasets - Combined data.csv'
    meta_df = pd.read_csv(meta)

    if meta == '/media/ssd2/hest/HEST_v0_0_1.csv':
        meta_df['image'] = [True for _ in range(len(meta_df))]
    #meta_df = meta_df[meta_df['Products'] == 'Spatial Gene Expression']
    meta_df = meta_df[meta_df['image'] == True]
    #meta_df = meta_df[meta_df['Products'] != 'HD Spatial Gene Expression']
    #meta_df = meta_df[meta_df['st_instrument'] != 'Xenium Analyzer']
    meta_df = meta_df[~meta_df['dataset_title'].isin(exclude_list)]
    #meta_df = meta_df[meta_df['st_technology'] != 'Xenium']
    #meta_df = meta_df[meta_df['dataset_title'] == 'Spatially resolved clonal copy number alterations in benign and malignant tissueJus']
    #meta_df = meta_df[meta_df['dataset_title'] == 'Spatially resolved clonal copy number alterations in benign and malignant tissueJus']
    #meta_df = meta_df[meta_df['dataset_title'] == 'FFPE Human Breast using the Entire Sample Area']
    #meta_df = meta_df[meta_df['check_image'] == "TRUE"] 
    #meta_df = meta_df[((meta_df['dataset_title'] == 'FFPE Human Breast using the Entire Sample Area') & (meta_df['subseries'] == 'Replicate 1')) |
    #             ((meta_df['dataset_title'] == 'FFPE Human Breast with Pre-designed Panel') & (meta_df['subseries'] == 'Tissue sample 1')) |
    #              ((meta_df['dataset_title'] == 'High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis [Xenium]') & (meta_df['subseries'] != 'Breast Cancer, Xenium In Situ Spatial Gene Expression Rep 2'))]
    #meta_df = meta_df[(meta_df['dataset_title'] == 'mousemodel_Heptablastoma spatial transcriptomics') & (meta_df['subseries'] == 'NEJ146-D')]
    #meta_df = meta_df[meta_df['dataset_title'] == 'mousemodel_Heptablastoma spatial transcriptomics' ]
    
    #meta_df = meta_df[(meta_df['dataset_title'] == "Spatiotemporal dynamics of molecular pathology in amyotrophic lateral sclerosis")] 
    """meta_df = meta_df[(meta_df['id'] == "NCBI758") | 
                      (meta_df['id'] == "TENX123") | 
                      (meta_df['id'] == "TENX66") | 
                      (meta_df['id'] == "TENX70") | 
                      (meta_df['id'] == "TENX81") | 
                      (meta_df['id'] == "MEND60") | 
                      (meta_df['id'] == "NCBI654") | 
                      (meta_df['id'] == "NCBI655") | 
                      (meta_df['id'] == "NCBI656") | 
                      (meta_df['id'] == "NCBI657") | 
                      (meta_df['id'] == "TENX125")]"""
    #meta_df = meta_df[meta_df['bigtiff'].notna() & meta_df['bigtiff'] != "TRUE"]
    #meta_df = meta_df[(meta_df['id'] == "TENX138")]
    #meta_df = meta_df[meta_df['id'].str.startswith('GIT')]
    #meta_df = meta_df[(meta_df['id'] == "NCBI783") | (meta_df['id'] == "NCBI785")]
    #meta_df = meta_df[(meta_df['id'] == "TENX137")]

    #meta_df = meta_df[(meta_df['use_train'] == 'TRUE')]

    dest = '/mnt/sdb1/paul/images'
    
    #df = pool_xenium_by_cell('/mnt/sdb1/paul/data/samples/xenium/FFPE Human Breast using the Entire Sample Area/Tissue sample 1', '/mnt/sdb1/paul/TENX95_cell_detection.geojson', 
    #                         0.2125)
    #df.to_parquet('TENX95_pool.parquet')
    
    
    #create_joined_gene_plots(meta_df, gene_plot=True)
    #adata = sc.read_h5ad('/mnt/sdb1/paul/images/adata/NCBI792.h5ad')
    #src_pixel_size = 0.988180746
    #tissue_mask = np.load('/mnt/sdb1/paul/fullres_mask.npy')
    #img = tifffile.imread('/mnt/sdb1/paul/images/pyramidal/NCBI792.tif')
    
    #mask_spots(adata, src_pixel_size, tissue_mask, 55.)
    """patchify(
        patch_save_dir='/mnt/sdb1/paul',
        gene_save_dir='/mnt/sdb1/paul',
        smoothed_save_dir='/mnt/sdb1/paul',
        adata=adata, 
        img=img, 
        src_pixel_size=src_pixel_size,
        name='test',
        patch_size_um=100.,
        tissue_mask=tissue_mask,
        target_pixel_size=0.5,
        verbose=1
    )"""

    meta_df = meta_df[(meta_df['dataset_title'] == "Multimodal decoding of human liver regeneration [st_human]")] 
    
    process_meta_df(meta_df)
    #meta_df = meta_df[(meta_df['dataset_title'] == "Spatial deconvolution of HER2-positive breast cancer delineates tumor-associated cell type interactions")] 
    #splits = meta_df.groupby('patient')['id'].agg(list).to_dict()
    #create_splits('/media/ssd2/hest/splits', splits, len(splits))
    #pyvips.tiffload('/media/ssd2/hest/pyramidal/TENX137.tif')
    #mask_and_patchify(meta_df[261:])
    """with Profile() as profile:
        mask_and_patchify(meta_df[0:1])
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.TIME)
            .print_stats()
        )"""
   #cProfile.run('mask_and_patchify(global_namespace["meta_df"][:1])')
    #mask_and_patchify(meta_df[:1])
    #top_k = get_k_mean_expressed_genes_from_df(meta_df, k=250, save_dir='BC1_250genes.json')
    
    
    # copy_processed_images(dest, meta_df, cp_spatial=False, cp_downscaled=False,)

    #open_fiftyone()
    
    #process_meta_df(meta_df[516:], save_spatial_plots=True, plot_genes=False)
    #process_meta_df(meta_df, save_spatial_plots=True, plot_genes=False)

    
    #process_meta_df(meta_df, save_spatial_plots=True, plot_genes=True)
    
    #pool_xenium_by_cell()
    
    #img = tifffile.imread('/mnt/sdb1/paul/data/samples/visium/Bern ST/20220401-2_7/20220401-2_7.ome.tif')
    #write_wsi2(img, '/mnt/sdb1/paul/test.tif')
    #create_joined_gene_plots(meta_df, gene_plot=True)
    #copy_processed_images(dest, meta_df, cp_spatial=False, cp_downscaled=False, cp_pyramidal=False, cp_pixel_vis=False)
    #copy_processed_images(dest, meta_df, cp_spatial=True, cp_downscaled=True, cp_pyramidal=True)
    
    #create_meta_release(meta_df, version.Version('0.0.1'))

if __name__ == "__main__":
    main()