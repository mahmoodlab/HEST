import pandas as pd
from src.hest.helpers import  create_meta_release, copy_processed_images, create_joined_gene_plots
from src.hest.readers import read_and_save, process_meta_df
from packaging import version
from PIL import Image
import tifffile
import openslide
import numpy as np
from openslide.deepzoom import DeepZoomGenerator

def main():

    exclude_list = [
        'Spatial Transcriptomic Experiment of Triple-Negative Breast Cancer PDX Model PIM001-P model treatment naive sample',
        'Visium Spatial Gene Expression of embryonic mouse brain at embryonic day 15.5',
        'Dissecting the melanoma ecosystem one cell at the time during immunotherapy',
        'Spatial Transcriptomics of human fetal liver',
        'Spatiotemporal mapping of immune and stem cell dysregulation after volumetric muscle loss',
        'Spatial transcriptomics profiling of the developing mouse embryo'
    ]

    meta = '/mnt/sdb1/paul/data/samples/ST H&E datasets - Combined data.csv'
    meta_df = pd.read_csv(meta)
    #meta_df = meta_df[meta_df['Products'] == 'Spatial Gene Expression']
    meta_df = meta_df[meta_df['image'] == True]
    meta_df = meta_df[meta_df['Products'] != 'HD Spatial Gene Expression']
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
    
    #create_joined_gene_plots(meta_df)
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
    meta_df = meta_df[(meta_df['id'] == "TENX138")]

    dest = '/mnt/sdb1/paul/images'
    
    
    # copy_processed_images(dest, meta_df, cp_spatial=False, cp_downscaled=False,)

    #open_fiftyone()
    
    #process_meta_df(meta_df[516:], save_spatial_plots=True, plot_genes=False)
    process_meta_df(meta_df, save_spatial_plots=True, plot_genes=False)
    #with tifffile.TiffFile('/mnt/sdb1/paul/data/samples/visium/Spatial transcriptomics landscape of non-communicable inflammatory skin diseases/Patient 28, 33 lesional psoriasis, lichen planus, rep 1 [21L008964]/processed/aligned_fullres_HE.tif') as tif:
        # Get the TIFF file header
    #    header = tif.pages[0].tags
    
    #process_meta_df(meta_df, save_spatial_plots=True)
    
    #img = tifffile.imread('/mnt/sdb1/paul/data/samples/visium/Bern ST/20220401-2_7/20220401-2_7.ome.tif')
    #write_wsi2(img, '/mnt/sdb1/paul/test.tif')
    #create_joined_gene_plots(meta_df, gene_plot=True)
    #copy_processed_images(dest, meta_df, cp_spatial=False, cp_downscaled=False, cp_pyramidal=False, cp_pixel_vis=False)
    #copy_processed_images(dest, meta_df, cp_spatial=True, cp_downscaled=True, cp_pyramidal=True)
    
    #create_meta_release(meta_df, version.Version('0.0.1'))

if __name__ == "__main__":
    main()