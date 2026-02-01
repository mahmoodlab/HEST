import json
import os
import pandas as pd
import scanpy
from tqdm import tqdm
from hest.utils import copy_processed, get_path_from_meta_row


id_list = [
    'TENX160',
    #'TENX187',
    #'TENX188',
]

df = pd.read_csv("/home/paul/Downloads/ST H&E datasets - 10XGenomics.csv")


df = df[df['id'].isin(id_list)]


def update_hest_meta(old_df, new_df, new_version: str):
    required_cols = [
        'dataset_title',
        'id',
        'image_filename',
        'organ',
        'disease_state',
        'oncotree_code',
        'species',
        'patient',
        'st_technology',
        'data_publication_date',
        'license',
        'study_link',
        'download_page_link1',
        'inter_spot_dist',
        'spot_diameter',
        'spots_under_tissue',
        'preservation_method',
        'nb_genes',
        'treatment_comment',
        'pixel_size_um_embedded',
        'pixel_size_um_estimated',
        'magnification',
        'fullres_px_width',
        'fullres_px_height',
        'tissue',
        'disease_comment',
        'subseries',
        'hest_version_added',
    ]
    
    new_rows = []
    for _, row in tqdm(new_df.iterrows()):
        path = get_path_from_meta_row(row)
        
        with open(os.path.join(path, 'processed', f'meta.json'), 'r') as f:
            meta = json.load(f)
        
        adata = scanpy.read_h5ad(os.path.join(path, 'processed', 'aligned_adata.h5ad'))
        nb_genes = len(adata.var_names)
            
        new_rows.append(
            {**{k: v for k, v in meta.items() if k in required_cols}, **{
                'hest_version_added': new_version,
                'image_filename': meta['id'] + '.tif',
                'nb_genes': nb_genes
            }}
        )
    new_df = pd.concat((pd.DataFrame(new_rows, columns=required_cols), old_df))
    return new_df
        

if __name__ == '__main__':
    copy_processed(
        '/media/paul/ssd2/HEST_results',
        df,
        cp_pyramidal=True,
        n_job=1,
        cp_meta=True,
        cp_adata=True,
        cp_cellvit=True,
        cp_downscaled=True,
        cp_patches=True,
        cp_spatial=True,
        cp_pixel_vis=True,
        cp_transcripts=True,
        cp_he_seg=True
    )