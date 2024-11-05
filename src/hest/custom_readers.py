import os
import subprocess

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .utils import (find_first_file_endswith, split_join_adata_by_col,
                    write_10X_h5)


def colon_atlas_to_adata(path):
    h5_path = find_first_file_endswith(path, 'filtered.h5ad')
    custom_adata = sc.read_h5ad(h5_path)
    #custom_adata.obs['pxl_col_in_fullres'] = custom_adata.obsm['spatial'][:, 0]
    #custom_adata.obs['pxl_row_in_fullres'] = custom_adata.obsm['spatial'][:, 1]
    custom_adata = custom_adata[custom_adata.obs['in_tissue'] == 1]
    return custom_adata

def heart_atlas_to_adata(path):
    h5_path = find_first_file_endswith(path, '.raw.h5ad')
    custom_data = sc.read_h5ad(h5_path)
    custom_data.obs['pxl_col_in_fullres'] = custom_data.obsm['spatial'][:, 0]
    custom_data.obs['pxl_row_in_fullres'] = custom_data.obsm['spatial'][:, 1]
    custom_data.obs.index = [idx.split('_')[1] for idx in custom_data.obs.index]
    return custom_data

def GSE238145_to_adata(path):
    counts_path = find_first_file_endswith(path, 'counts.txt')
    coords_path = find_first_file_endswith(path, 'coords.txt')
    
    coords_df = pd.read_csv(coords_path, index_col=0, sep='\t')
    coords_df = coords_df.rename(columns={
        'tissue': 'in_tissue',
        'row': 'array_row',
        'col': 'array_col',
        'imagerow': 'pxl_row_in_fullres',
        'imagecol': 'pxl_col_in_fullres',
    })
    
    counts_df = pd.read_csv(counts_path, index_col=0, sep='\t')
    counts_df = counts_df.transpose()
    counts_df.index = counts_df.index.str.replace('.', '-')
    
    sel = counts_df.merge(coords_df, left_index=True, right_index=True).index
    coords_df = coords_df.loc[sel]
    
    adata = sc.AnnData(counts_df)
    
    adata.obsm['spatial'] = np.column_stack((coords_df['pxl_col_in_fullres'], coords_df['pxl_row_in_fullres']))
    return adata


def join_object_to_adatas_GSE214989(path):
    adata = sc.read_10x_h5(path)
    sampleIDS = ['_1', '_2', '_3']
    for sampleID in sampleIDS:
        my_adata = adata.copy()
        df = my_adata.obs
        df = df[df.index.str.endswith(sampleID)]
        new_df = my_adata.to_df().loc[df.index]
        new_df.index = [idx[:-2] for idx in new_df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'{sampleID}_filtered_feature_bc_matrix.h5'))


def join_object_to_adatas_GSE171351(path):
    adata = sc.read_h5ad(path)
    sampleIDS = ['A1', 'B1', 'C1', 'D1']
    for sampleID in sampleIDS:
        my_adata = adata.copy()
        df = my_adata.obs#.reset_index(drop=True)
        df = df[df['sampleID'] == sampleID]
        new_df = my_adata.to_df().loc[df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        new_adata.obs = my_adata.obs[my_adata.obs['sampleID'] == sampleID]
        
        
        new_adata.uns['spatial'] = my_adata.uns['spatial'][sampleID]
        
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'{sampleID}_filtered_feature_bc_matrix.h5'))



def align_dev_human_heart(raw_counts_path, spot_coord_path, exp_name):
    EXP_ORDER = ['ST_Sample_4.5-5PCW_1', 'ST_Sample_4.5-5PCW_2', 
                 'ST_Sample_4.5-5PCW_3', 'ST_Sample_4.5-5PCW_4',
                 'ST_Sample_6.5PCW_1', 'ST_Sample_6.5PCW_2', 
                 'ST_Sample_6.5PCW_3', 'ST_Sample_6.5PCW_4',
                 'ST_Sample_6.5PCW_5', 'ST_Sample_6.5PCW_6',
                 'ST_Sample_6.5PCW_7', 'ST_Sample_6.5PCW_8',
                 'ST_Sample_6.5PCW_9', 'ST_Sample_9PCW_1',
                 'ST_Sample_9PCW_2', 'ST_Sample_9PCW_3',
                 'ST_Sample_9PCW_4', 'ST_Sample_9PCW_5',
                 'ST_Sample_9PCW_6']
    EXP_MAP = {key: value for key, value in zip(EXP_ORDER, np.arange(19) + 1)}
    
    spot_coords = pd.read_csv(spot_coord_path, sep='\t')
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    
    # select 
    exp_id = EXP_MAP[exp_name]
    col_mask = [col for col in raw_counts.columns if col.startswith(f'{exp_id}x')]
    raw_counts = raw_counts[col_mask]
    raw_counts = raw_counts.transpose()
    spot_coords.index = [str(exp_id) + 'x' for _ in range(len(spot_coords))] + spot_coords['x'].astype(str) + ['x' for _ in range(len(spot_coords))] + spot_coords['y'].astype(str)
    
    merged = pd.merge(raw_counts, spot_coords, left_index=True, right_index=True, how='inner')
    raw_counts = raw_counts.reindex(merged.index)
    adata = sc.AnnData(raw_counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    adata.obsm['spatial'] = matrix
    
    return adata


def GSE206391_split_h5(path):
    adata = sc.read_h5ad(path)
    a = 1
    library_ids = np.unique(adata.obs['library_id'])
    for library_id in library_ids:
        old_library_id = library_id
        library_id = "_".join(library_id.split('_')[:2])
        new_adata = adata.copy()
        adata.obs['int_index'] = np.arange(len(adata.obs))
        df = adata.obs[adata.obs['library_id'] == old_library_id]
        
        new_df = adata.to_df().iloc[df['int_index']]
        new_df.index = [idx + '-1' for idx in new_df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        new_adata.obsm['spatial'] = adata.obsm['spatial'][df['int_index'].values.astype(int)]
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'{library_id}_filtered_feature_bc_matrix.h5'))
        
        
def _GSE206391_copy_dir(path):
    for dir in os.listdir(path):
        if dir.endswith('filtered_feature_bc_matrix.h5'):
            whole_path = os.path.join(path, dir)
            if '21L' in dir:
                sample_name = dir.split('_filtered_feature_bc_matrix.h5')[0].split('_')[1]
            else:
                sample_name = dir.split('_filtered_feature_bc_matrix.h5')[0]
            param = f'mv "{whole_path}" "{path}/"*{sample_name}*'
            subprocess.Popen(param, shell=True)

def GSE234047_to_adata(path):
    path = find_first_file_endswith(path, '_counts.csv')
            
    df = pd.read_csv(path)
    df.index = df['barcode']
    columns_drop = ['barcode', 'prediction_celltype', 'Bipolar', 'Cone', 'Endothelial', 'Fibroblast', 'Immune', 'Interneuron', 'Melanocyte', 'Muller.Astrocyte', 'Pericyte.SMC', 'RGC', 'Rod', 'RPE.x', 'Schwann', 'res_ss', 'region', 'tissue', 'percent_CNV', 'image']
    
    df = df.drop(columns_drop, axis=1)
    
    df.index = [s.split('_')[1].split('-')[0] + '-1' for s in df.index]
    
    adata = sc.AnnData(df)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X)
    return adata


def GSE184384_to_h5(path):
    
    mex_path = find_first_file_endswith(path, 'mex')
    adata = sc.read_10x_mtx(mex_path)
    
    
            
    df = pd.read_csv(path)
    df.index = df['barcode']
    columns_drop = ['barcode', 'prediction_celltype', 'Bipolar', 'Cone', 'Endothelial', 'Fibroblast', 'Immune', 'Interneuron', 'Melanocyte', 'Muller.Astrocyte', 'Pericyte.SMC', 'RGC', 'Rod', 'RPE.x', 'Schwann', 'res_ss', 'region', 'tissue', 'percent_CNV', 'image']
    
    df = df.drop(columns_drop, axis=1)
    
    df.index = [s.split('_')[1].split('-')[0] + '-1' for s in df.index]
    
    adata = sc.AnnData(df)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X)
    return adata


def GSE180128_to_adata(path):
    path = find_first_file_endswith(path, '.csv')
    df = pd.read_csv(path)
    df.index = df['Unnamed: 0']
    df = df.drop(['Unnamed: 0'], axis=1)
    adata = sc.AnnData(df)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X)
    return adata    


def GSE184369_to_h5(path):
    adata = sc.read_10x_mtx(path)
    
    
    df = pd.read_csv(path)
    df.index = df['barcode']
    columns_drop = ['barcode', 'prediction_celltype', 'Bipolar', 'Cone', 'Endothelial', 'Fibroblast', 'Immune', 'Interneuron', 'Melanocyte', 'Muller.Astrocyte', 'Pericyte.SMC', 'RGC', 'Rod', 'RPE.x', 'Schwann', 'res_ss', 'region', 'tissue', 'percent_CNV', 'image']
    
    df = df.drop(columns_drop, axis=1)
    
    df.index = [s.split('_')[1].split('-')[0] + '-1' for s in df.index]
    
    adata = sc.AnnData(df)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X)
    return adata


def GSE167096_to_adata(path):
    symbol_path = find_first_file_endswith(path, 'symbol.txt')

    matrix = pd.read_csv(symbol_path, sep='\t')
    matrix.index = matrix['Symbol']
    matrix = matrix.transpose().iloc[1:]

    adata = sc.AnnData(matrix)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X.astype(int))
    
    return adata


def GSE203165_to_adata(path):
    path = find_first_file_endswith(path, 'raw_counts.txt')
    matrix = pd.read_csv(path, sep='\t', index_col=0)
    matrix = matrix.transpose()
    adata = sc.AnnData(matrix)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X.astype(int))
    return adata


def GSE205707_split_to_h5ad(path):
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
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
        except:
            sample_adata.__dict__['_raw'].__dict__['_var'] = sample_adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
            
            
def align_eval_qual_dataset(raw_counts_path, spot_coord_path):
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    spot_coords = pd.read_csv(spot_coord_path, sep='\t')
    
    spot_coords.index = spot_coords['x'].astype(str) + ['x' for _ in range(len(spot_coords))] + spot_coords['y'].astype(str)
    
    merged = pd.merge(raw_counts, spot_coords, left_index=True, right_index=True, how='inner')
    raw_counts = raw_counts.reindex(merged.index)
    adata = sc.AnnData(raw_counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    adata.obsm['spatial'] = matrix
    
    return adata


def align_her2(path, raw_count_path):
    selection_path = find_first_file_endswith(path, 'selection.tsv')
    spot_coords = pd.read_csv(selection_path, sep='\t')
    raw_counts = pd.read_csv(raw_count_path, sep='\t', index_col=0)
    
    spot_coords.index = spot_coords['x'].astype(str) + ['x' for _ in range(len(spot_coords))] + spot_coords['y'].astype(str)
    
    merged = pd.merge(raw_counts, spot_coords, left_index=True, right_index=True, how='inner')
    raw_counts = raw_counts.reindex(merged.index)
    adata = sc.AnnData(raw_counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    adata.obsm['spatial'] = matrix
    adata.obs['array_row'] = [int(s.split('x')[0]) for s in adata.obs.index]
    adata.obs['array_col'] = [int(s.split('x')[1]) for s in adata.obs.index]
    adata.obs['pxl_row_in_fullres'] = col2
    adata.obs['pxl_col_in_fullres'] = col1
    adata.obs['in_tissue'] = True
              
    return adata
        


def infer_row_col_from_barcodes(barcodes_df, adata):
    barcode_path = './assets/barcode_coords/visium-v1_coordinates.txt'
    barcode_coords = pd.read_csv(barcode_path, sep='\t', header=None)
    barcode_coords = barcode_coords.rename(columns={
        0: 'barcode',
        1: 'array_col',
        2: 'array_row'
    })
    barcode_coords['barcode'] += '-1'
    
    # space rangers provided barcode coords are 1 indexed whereas alignment file are 0 indexed
    barcode_coords['array_col'] -= 1
    barcode_coords['array_row'] -= 1
    
    barcodes_df['barcode'] = barcodes_df.index

    spatial_aligned = pd.merge(barcodes_df, barcode_coords, on='barcode', how='inner')

    spatial_aligned.index = spatial_aligned['barcode']

    spatial_aligned = spatial_aligned[['array_row', 'array_col']]

    spatial_aligned = spatial_aligned.reindex(adata.obs.index)
    return spatial_aligned



def GSE217828_to_adata(path):
    raw_counts_path = (path, 'raw_count.csv')
    raw_counts = pd.read_csv(raw_counts_path)
    raw_counts = raw_counts.transpose()
    raw_counts.index = [idx.split('_')[1].replace('.', '-') for idx in raw_counts.index]
    meta_path = (path, 'meta_data.csv')
    meta = pd.read_csv(meta_path)
    meta.index = [idx.split('_')[1] for idx in meta.index]
    #meta.index = meta['nCount_SCT']
    matrix = np.column_stack((meta['Coord_x_slide'], abs(meta['Coord_y_slide'])))
    
    
    raw_counts = raw_counts.reindex(meta.index)
    
    adata = sc.AnnData(raw_counts)
    adata.var['feature_types'] = ['Gene Expression' for _ in range(len(adata.var))]
    adata.var['genome'] = ['Unspecified' for _ in range(len(adata.var))]
    adata.X = sparse.csr_matrix(adata.X.astype(int))
    
    adata.obsm['spatial'] = matrix
    #adata.obs['']
    
    # TODO infer row and col from template
    spatial_aligned = infer_row_col_from_barcodes(adata.obs, adata)
    adata.obs['array_row'] = spatial_aligned['array_row']
    adata.obs['array_col'] = spatial_aligned['array_col']
    adata.obs['pxl_col_in_fullres'] = matrix[:, 1]
    adata.obs['pxl_row_in_fullres'] = matrix[:, 0]
    adata.obs['in_tissue'] = [True for _ in range(len(adata.obs))]
    
    return adata
    
    
    
def GSE236787_split_to_h5(path):
    adata = sc.read_10x_h5(path)
    sampleIDS = ['-1', '-2', '-3', '-4']
    for sampleID in sampleIDS:
        my_adata = adata.copy()
        df = my_adata.obs#.reset_index(drop=True)
        df = df.loc[[i for i in df.index.values if i.endswith(sampleID)]]
        new_df = my_adata.to_df().loc[df.index]
        new_df.index = [s.split('-')[0] + '-1' for s in new_df.index]
        new_adata = sc.AnnData(new_df, var=adata.var)
        new_adata.var['feature_types'] = ['Gene Expression' for _ in range(len(new_adata.var))]
        new_adata.var['genome'] = ['Unspecified' for _ in range(len(new_adata.var))]
        new_adata.X = sparse.csr_matrix(new_adata.X)
        new_adata.obs = df
        new_adata.obs.index = new_df.index
        
        write_10X_h5(new_adata, os.path.join(os.path.dirname(path), f'N{sampleID}filtered_feature_bc_matrix.h5'))

    return df


def _ST_spot_to_pixel(x, y, img):
    ARRAY_WIDTH = 6200.0
    ARRAY_HEIGHT = 6600.0
    SPOT_SPACING = ARRAY_WIDTH/(31+1)
    
    pixelDimX = (SPOT_SPACING*img.shape[1])/(ARRAY_WIDTH)
    pixelDimY = (SPOT_SPACING*img.shape[0])/(ARRAY_HEIGHT)
    return (x-1)*pixelDimX,(y-1)*pixelDimY


def align_ST_counts_with_transform(raw_counts_path, transform_path):
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    with open(transform_path) as file:
        aff_transform = np.array(file.read().split(' '))
        aff_transform = aff_transform.reshape((3, 3)).astype(float).T
    xy = np.array([[idx.split('x')[0], idx.split('x')[1], 1] for idx in raw_counts.index]).astype(float)
    xy_aligned = (aff_transform @ xy.T).T
    adata = sc.AnnData(raw_counts)
    matrix = xy_aligned[:, :2]
    adata.obsm['spatial'] = matrix
    
    return adata


def raw_counts_to_pixel(raw_counts_df, img):
    spot_coords = []
    for col in raw_counts_df.columns:
        tup = col.split('_')
        x, y = _ST_spot_to_pixel(float(tup[0]), float(tup[1]), img)
        spot_coords.append([x, y])
    return np.array(spot_coords)


def raw_count_to_adata(raw_count_path):
    matrix = pd.read_csv(raw_count_path, sep=',')
    matrix.index = matrix['Gene']
    matrix = matrix.transpose().iloc[1:]

    adata = sc.AnnData(matrix)

    return adata


def GSE144239_to_adata(raw_counts_path, spot_coord_path):
    import scanpy as sc
    
    raw_counts = pd.read_csv(raw_counts_path, sep='\t', index_col=0)
    spot_coord = pd.read_csv(spot_coord_path, sep='\t')
    spot_coord.index = spot_coord['x'].astype(str) + ['x' for _ in range(len(spot_coord))] + spot_coord['y'].astype(str)
    merged = pd.merge(spot_coord, raw_counts, left_index=True, right_index=True)
    raw_counts = raw_counts.reindex(merged.index)
    adata = sc.AnnData(raw_counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    adata.obsm['spatial'] = matrix
    return adata


def ADT_to_adata(img_path, raw_counts_path):
    import scanpy as sc
    
    basedir = os.path.dirname(img_path)
    # combine spot coordinates into a single dataframe
    pre_adt_path= find_first_file_endswith(basedir, 'pre-ADT.tsv')
    post_adt_path = find_first_file_endswith(basedir, 'postADT.tsv')
    if post_adt_path is None:
        post_adt_path = find_first_file_endswith(basedir, 'post-ADT.tsv')
    counts = pd.read_csv(raw_counts_path, index_col=0, sep='\t')
    pre_adt = pd.read_csv(pre_adt_path, sep='\t')
    post_adt = pd.read_csv(post_adt_path, sep='\t')
    merged_coords = pd.concat([pre_adt, post_adt], ignore_index=True)
    merged_coords.index = [str(x) + 'x' + str(y) for x, y in zip(merged_coords['x'], merged_coords['y'])]
    merged = pd.merge(merged_coords, counts, left_index=True, right_index=True, how='inner')
    counts = counts.reindex(merged.index)
    adata = sc.AnnData(counts)
    col1 = merged['pixel_x'].values
    col2 = merged['pixel_y'].values
    matrix = (np.vstack((col1, col2))).T
    adata.obsm['spatial'] = matrix
    return adata