import pandas as pd
from src.hiSTloader.helpers import  create_meta_release, process_meta_df, copy_processed_images
from src.hiSTloader.readers import read_and_save
from packaging import version


def main():

    exclude_list = [
        'Spatial Transcriptomic Experiment of Triple-Negative Breast Cancer PDX Model PIM001-P model treatment naive sample',
        'Visium Spatial Gene Expression of embryonic mouse brain at embryonic day 15.5',
        'Dissecting the melanoma ecosystem one cell at the time during immunotherapy',
        'Spatial Transcriptomics of human fetal liver'
    ]

    meta = '/mnt/sdb1/paul/data/samples/ST H&E datasets - Combined data.csv'
    meta_df = pd.read_csv(meta)
    #meta_df = meta_df[meta_df['Products'] == 'Spatial Gene Expression']
    #meta_df = meta_df[meta_df['id'].str.startswith('MEND')]
    meta_df = meta_df[meta_df['image'] == True]
    meta_df = meta_df[meta_df['Products'] != 'HD Spatial Gene Expression']
    meta_df = meta_df[meta_df['st_instrument'] == 'Xenium Analyzer']
    meta_df = meta_df[~meta_df['dataset_title'].isin(exclude_list)]
    #meta_df = meta_df[meta_df['dataset_title'] == 'Spatially resolved clonal copy number alterations in benign and malignant tissueJus']
    #meta_df = meta_df[meta_df['dataset_title'] == 'Spatially resolved clonal copy number alterations in benign and malignant tissueJus']
    #meta_df = meta_df[meta_df['dataset_title'] == 'FFPE Human Breast using the Entire Sample Area']
    #meta_df = meta_df[meta_df['check_image'] == "TRUE"] 
    #meta_df = meta_df[((meta_df['dataset_title'] == 'FFPE Human Breast using the Entire Sample Area') & (meta_df['subseries'] == 'Replicate 1')) |
    #              ((meta_df['dataset_title'] == 'FFPE Human Breast with Pre-designed Panel') & (meta_df['subseries'] == 'Tissue sample 1')) |
    #              ((meta_df['dataset_title'] == 'High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis [Xenium]') & (meta_df['subseries'] != 'Breast Cancer, Xenium In Situ Spatial Gene Expression Rep 2'))]
    # meta_df = meta_df[(meta_df['dataset_title'] == 'Bern ST') & (meta_df['subseries'] == '20220401-2_5')]
    #meta_df = meta_df[((meta_df['dataset_title'] == 'Human Breast Cancer (Block A Section 2)') & (meta_df['subseries'] != 'Replicate 1'))]
    
    #create_joined_gene_plots(meta_df)
    

    dest = '/mnt/sdb1/paul/images'
    #copy_processed_images(dest, meta_df, cp_spatial=False, cp_downscaled=False)

    #open_fiftyone()
    
    #process_meta_df(meta_df, save_spatial_plots=True, plot_genes=False)
    

    process_meta_df(meta_df, save_spatial_plots=True) #230
    
    copy_processed_images(dest, meta_df, cp_spatial=True, cp_downscaled=True)
    
    create_meta_release(meta_df, version.Version('0.0.1'), update_pixel_size=True)
        

if __name__ == "__main__":
    main()