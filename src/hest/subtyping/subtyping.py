from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from hest.subtyping.atlas import (get_atlas_from_name, get_cells_with_clusters,
                                  sc_atlas_factory)
from hest.subtyping.atlas_matchers import matcher_factory
from hest.utils import get_path_from_meta_row

level = 'predicted.celltypel1'


def assign_cell_types(cell_adata, atlas_name, name, method='tangram', full_atlas=False, organ=None, **matcher_kwargs) -> sc.AnnData:
    matcher = matcher_factory(method)
    
    
    if organ is not None:
        atlas_cells = sc_atlas_factory(organ)
    else:
        atlas_cells = get_atlas_from_name(atlas_name)()
        
    if not full_atlas:
        atlas_cells = atlas_cells.get_downsampled()
    else:
        atlas_cells = atlas_cells.get_full()
    
    preds = matcher.match_atlas(
        name, 
        cell_adata, 
        atlas_cells, 
        **matcher_kwargs)
    return preds
    

def assign_cell_types_hest(meta_df, method='tangram'):
    for _, row in meta_df.iterrows():
        path = get_path_from_meta_row(row)
        organ = row['Organ']
        
        cell_adata = get_cells_with_clusters(path, cluster_path=os.path.join(path, 'analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv'), k=None)
        assign_cell_types(cell_adata, organ, method=method)
            

def place_in_right_folder(path, rename=False):
    paths = os.listdir(path)
    ids = np.unique([path.split('.')[0] for path in paths])
    for id in ids:
        os.makedirs(os.path.join(path, id), exist_ok=True)
    for f in paths:
        src = os.path.join(path, f)
        
        if not os.path.isfile(src):
            continue
        
        name = f.split('.')[0]
        if rename:
            if 'barcodes' in f:
                f = 'barcodes.tsv.gz'
            elif 'features' in f or 'genes' in f:
                f = 'features.tsv.gz'
            elif 'matrix' in f:
                f = 'matrix.mtx.gz' 
        dst = os.path.join(path, name, f)
        shutil.move(src, dst)
        
        
def join_MEX(dir):
    import scanpy as sc
    joined_adata = None
    for f in tqdm(os.listdir(dir)):
        path = os.path.join(dir, f)
        if os.path.isdir(path):
            adata = sc.read_10x_mtx(path)
            if joined_adata is None:
                joined_adata = adata
            else:
                joined_adata = joined_adata.concatenate(adata, join='outer')
    return joined_adata
    


xenium_cell_types_map = {
    'Adipocytes': 'Stromal',
    'B Cells': 'B-cells',
    'CD163+ Macrophage': 'Myeloid',
    'CD83+ Macrophage': 'Myeloid',
    'CTLA4+ T Cells': 'T-cells',
    'DST+ Myoepithelial': 'Normal Epithelial',
    'ESR1+ Epithelial': 'Normal Epithelial',
    'Endothelial': 'Endothelial',
    'ITGAX+ Macrophage': 'Myeloid',
    'Mast Cells': 'Myeloid',
    'Not Plotted': 'NA',
    'OPRPN+ Epithelial': 'Normal Epithelial',
    'PIGR+ Epithelial': 'Normal Epithelial',
    'Plasma Cells': 'B-cells',
    'Plasmacytoid Dendritic': 'B-cells',
    'Stromal Normal': 'Stromal',
    'TRAC+ Cells': 'T-cells',
    'Transitional Cells': 'NA',
    'Tumor': 'Cancer Epithelial',
    'Tumor Associated Stromal': 'Stromal',
    'B_Cells': 'B-cells',
    'CD4+_T_Cells': 'T-cells',
    'CD8+_T_Cells': 'T-cells',
    'DCIS_1': 'Cancer Epithelial',
    'DCIS_2': 'Cancer Epithelial',
    'IRF7+_DCs': 'Myeloid',
    'Invasive_Tumor': 'Cancer Epithelial',
    'LAMP3+_DCs': 'Myeloid',
    'Macrophages_1': 'Myeloid',
    'Macrophages_2': 'Myeloid',
    'Mast_Cells': 'Myeloid',
    'Myoepi_ACTA2+': 'Normal Epithelial',
    'Myoepi_KRT15+': 'Normal Epithelial',
    'Perivascular-Like': 'Stromal',
    'Prolif_Invasive_Tumor': 'Cancer Epithelial',
    'Stromal': 'Stromal',
    'Stromal_&_T_Cell_Hybrid': 'NA',
    'T_Cell_&_Tumor_Hybrid': 'NA',
    'Unlabeled': 'NA',
    'NK Cells': 'T-cells',
    'Macrophage 1': 'Myeloid',
    'Macrophage 2': 'Myeloid',
    'Mast Cells': 'Myeloid',
    'Plasmablast': 'B-cells',
    'Invasive Tumor': 'Cancer Epithelial',
    'Undefined': 'NA',
    'T Cells': 'T-cells',
    'DCIS': 'Cancer Epithelial',
    'ACTA2+ Myoepithelial': 'Normal Epithelial',
    'KRT15+ Myoepithelial': 'Normal Epithelial'
}

breast3_cell_types_map = {
    'basal na': 'Normal Epithelial',
    'bcells na': 'B-cells',
    'fibroblasts na': 'Stromal',
    'lumhr na': 'Normal Epithelial',
    'lumsec na': 'Normal Epithelial',
    'lumsec proliferating': 'Cancer Epithelial',
    'lymphatic na': 'Endothelial',
    'myeloid na': 'Myeloid',
    'myeloid proliferating': 'Myeloid',
    'pericytes na': 'Stromal',
    'tcells na': 'T-cells',
    'tcells proliferating': 'T-cells',
    'vascular na': 'Endothelial'
}

def eval_cell_type_assignment(pred, path_gt, map_pred, map_gt, name, key='cell_type_pred'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    
    if isinstance(pred, str):
        df1 = pd.read_csv(pred)
    else:
        df1 = pred
    df2 = pd.read_csv(path_gt)
    if not isinstance(df2['Barcode'].iloc[0], str):
        df2['Barcode'] = df2['Barcode'].astype(int).astype(str)
    
    merged = df1.merge(df2, left_index=True, right_on='Barcode', how='inner')
    
    merged['Mapped'] = [map_gt[x] for x in merged['Cluster'].values]
    
    
    
    mask_NA = merged['Mapped'] == 'NA'
    merged = merged[~mask_NA]
     
    mapped_pred = [map_pred[x] for x in merged[key].values]
    mapped_gt = [map_gt[x] for x in merged['Cluster'].values]
    
    labels = sorted(list(set(mapped_gt) | set(mapped_pred)))
    cm = confusion_matrix(mapped_gt, mapped_pred, labels=labels)
    
    balanced_acc = round(balanced_accuracy_score(mapped_gt, mapped_pred), 4)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion matrix, cell type prediction (balanced_acc={balanced_acc})')
    
    plt.tight_layout()
    
    plt.savefig(name + 'confusion_matrix.jpg', dpi=150)
    
    
def eval_all_cell_type_assignments(meta_df):
    for id in meta_df['id']:
        for method in ['tangram', 'harmony']:
            eval_cell_type_assignment(f'cell_type_preds/{method}_{id}k=10.csv', f'cell_type_preds/gt_{id}.csv', breast3_cell_types_map, xenium_cell_types_map, name=f'{id}_{method}_')
    
