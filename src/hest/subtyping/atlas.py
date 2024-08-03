from __future__ import annotations

import os
from abc import abstractmethod

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ATLAS_PATH = 'atlas'

def downsample_and_save(X, cell_types, var_names, filename, level='cell_types'):
    import scanpy as sc
    sub_adata, sub_cell_types = downsample_atlas(X, cell_types)    
    new_adata = sc.AnnData(sub_adata, obs=pd.DataFrame(sub_cell_types.values, columns=[level]))
    new_adata.var_names = var_names

    new_adata.write_h5ad(filename)
    
def downsample_atlas(X, cell_types_arr):
    print('downsample atlas...')
    cell_types_arr = pd.Series(cell_types_arr)
    cell_types, counts = np.unique(cell_types_arr, return_counts=True)
    all_idx = []
    for i in range(len(cell_types)):
        cell_type = cell_types[i]
        idx = cell_types_arr[cell_types_arr == cell_type].index
        idx = list(np.random.choice(idx, min(1000, counts[i])))
        all_idx += idx
    X = X[all_idx]
    cell_types_arr = cell_types_arr[all_idx]
    return X, cell_types_arr


class SCAtlas:
    
    type_map = None
    name = None
    
    def process(self) -> None:
        full_adata = self.process_imp()
        
        if self.type_map is not None:
            logger.info('remap cell types...')
            full_adata.obs['cell_types'] = [self.type_map[x] for x in full_adata.obs['cell_types']]
        
        sub = 1000
        full_adata.write_h5ad(os.path.join(ATLAS_PATH, self.name, 'full.h5ad'))
        save_dir = os.path.join('atlas', self.name)
        
        levels = ['cell_types']
        if 'cell_types_lvl2' in full_adata.obs.columns:
            levels += ['cell_types_lvl2']
        
        for level in levels:
            name = os.path.join(save_dir, f'sub_{sub}')
            if level != 'cell_types':
                name += '_' + level
            downsample_and_save(full_adata.X, full_adata.obs[level].values, full_adata.var_names, name + '.h5ad', level=level)
    
        
        
    @abstractmethod
    def process_imp(self):
        pass
    
    def get_downsampled(self, level='cell_types') -> sc.AnnData:
        name = 'sub_1000'
        if level != 'cell_types':
            name += f'_{level}'
        path = os.path.join(ATLAS_PATH, self.name, name + '.h5ad')
        adata = self._get_or_process(path)
        return adata
        
    def get_full(self) -> sc.AnnData:
        path = os.path.join(ATLAS_PATH, self.name, 'full.h5ad')
        adata = self._get_or_process(path)
        return adata
        
    def _get_or_process(self, path) -> sc.AnnData:
        import scanpy as sc
        if not os.path.exists(path):
            logger.info(f"{path} doesn't exist, processing atlas...")
            self.process()
        adata = sc.read_h5ad(path)
        return adata

class HeartAtlas(SCAtlas):
    
    def __init__(self):
        self.name = 'heart'
    
    def process_imp(self):
        import scanpy as sc
        adata = sc.read_h5ad('/mnt/ssd/paul/ST-histology-loader/atlas/heart/Global_lognormalised.h5ad')
        adata.obs = adata.obs[['cell_type']]
        adata.obs = adata.obs.rename(columns={
            'cell_type': 'cell_types'
        })
        
        return adata


def find_common_genes(dfs):
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns.intersection_update(df.columns)

    common_columns = list(common_columns)
    return common_columns


class MEL2Atlas(SCAtlas):

    def __init__(self):
        self.name = 'MEL2'   
        self.type_map = {
            'B.cell': 'B-cells',
            'CAF': 'CAFs',
            'Endo.': 'Endothelial cells',
            'Macrophage': 'Macrophages',
            'Mal': 'Epithelial cancer cells',
            'NK': 'NK-cells',
            'T.CD4': 'T-cells',
            'T.CD8': 'T-cells',
            'T.cell': 'T-cells',
            'Endothelial.cell': 'Endothelial cells'
        }
        
        self.coarse_map = {
            'B-cells': 'Lymphocytes',
            'CAFs': 'Fibroblasts',
            'Endothelial cells': 'Endothelial cells',
            'Macrophages': 'Macrophages',
            'NK-cells': 'Lymphocytes',
            'T-cells': 'Lymphocytes'
        }
        
        
    def process_imp(self):
        import scanpy as sc

        #path_exp = 'atlas/MEL2/expression'
        #dfs = []
        #for exp in tqdm(os.listdir(path_exp)):
        #    path = os.path.join(path_exp, exp)
        #    df = pd.read_csv(path, sep='\t', index_col=0).transpose()
        #    dfs.append(df)
        #     
        #counts = pd.concat(dfs)
        #counts = counts.dropna(axis=1, how='any')
        #counts.to_parquet('atlas/MEL2/exp.parquet')
        counts = pd.read_parquet('atlas/MEL2/exp.parquet')
        
        meta1 = pd.read_csv('atlas/MEL2/tumors.mal_tsne_anno.txt', sep='\t', index_col=0)
        meta1 = meta1.iloc[1:]
        meta2 = pd.read_csv('atlas/MEL2/tumors.nonmal_tsne_anno.txt', sep='\t', index_col=0)
        meta2 = meta2.iloc[1:]
        meta = pd.concat((meta1, meta2))
        merged = counts.merge(meta, left_index=True, right_index=True, how='left')[['cell.type']]
        merged = merged.rename(columns={
            'cell.type': 'cell_types'
        })
        adata = sc.AnnData(counts)
        adata.obs = merged
        adata = adata[merged.dropna().index]
        return adata
        

class TonsilAtlas(SCAtlas):
    
    def __init__(self):
        self.name = 'tonsil'
        self.type_map = {
            'Activated NBC': 'Activated naive B-cells',
            'CD4 T': 'CD4 T-cells',
            'CD8 T': 'CD8 T-cells',
            'DC': 'Dendritic cells',
            'DN': 'Double negative T-cells',
            'FDC': 'Follicular dendritic cells',
            'GCBC': 'GC B-cells',
            'Granulocytes': 'Granulocytes',
            'ILC': 'Innate lymphoid cells',
            'MBC': 'Memory B-cells',
            'Mast': 'Mast cells',
            'Mono/Macro': 'Macrophages',
            'NA': 'NA',
            'NBC': 'Naive B-cells',
            'NK': 'NK-cells',
            'Naive CD4 T': 'Naive CD4 T-cells',
            'Naive CD8 T': 'Naive CD8 T-cells',
            'PC': 'Plasma cells',
            'PDC': 'Plasmacytoid dendritic cells',
            'cycling FDC': 'Cycling follicular dendritic cells',
            'cycling T': 'Cycling T-cells',
            'cycling myeloid': 'Cycling myeloid',
            'epithelial': 'Epithelial normal cells',
            'preB/T': 'Pre B/T-cells'
        }
        
        self.coarse_map = {
            'Activated naive B-cells': 'Lymphocytes',
            'CD4 T-cells': 'Lymphocytes',
            'CD8 T-cells': 'Lymphocytes',
            'Cycling T-cells': 'Lymphocytes',
            'Cycling follicular dendritic cells': 'NA',
            'Cycling myeloid': 'NA',
            'Dendritic cells': 'Dendritic cells',
            'Double negative T-cells': 'Lymphocytes',
            'Epithelial normal cells': 'Epithelial normal cells',
            'Follicular dendritic cells': 'NA',
            'GC B-cells': 'Lymphocytes',
            'Granulocytes': 'Granulocytes',
            'Innate lymphoid cells': 'Lymphocytes',
            'Macrophages': 'Macrophages',
            'Mast cells': 'Mast cells',
            'Memory B-cells': 'Lymphocytes',
            'NA': 'NA',
            'NK-cells': 'Lymphocytes',
            'Naive B-cells': 'Lymphocytes',
            'Naive CD4 T-cells': 'Lymphocytes',
            'Naive CD8 T-cells': 'Lymphocytes',
            'Plasma cells': 'Plasma cells',
            'Plasmacytoid dendritic cells': 'NA',
            'Pre B/T-cells': 'Lymphocytes'
        }
    
    def process_imp(self):
        import scanpy as sc
        adata = sc.read_h5ad('atlas/tonsil/tonsil.h5ad')
        adata.obsm = None
        adata.obs = adata.obs[['annotation_figure_1']]
        adata.obs = adata.obs.rename(columns={
            'annotation_figure_1': 'cell_types'
        })
        
        return adata

class IDCAtlas(SCAtlas):
    
    def __init__(self):
        self.name = 'IDC'
        
    
    def process_imp(self):
        import scanpy as sc
        
        ids = ['NCBI783', 'NCBI784', 'NCBI785']
        sub_adatas = []
        for id in ids:
            annot = pd.read_csv(f'atlas/IDC/gt_{id}.csv')
            if id == 'NCBI784' or id == 'NCBI785':
                annot['Barcode'] = annot['Barcode'].astype(int).astype(str)
            adata = sc.read_10x_h5(f'atlas/IDC/{id}.h5')
            merged = adata.obs.merge(annot, left_index=True, right_on='Barcode', how='left')[['Cluster']]
            adata.obs = merged
            adata.obs = adata.obs.rename(columns={
                'Cluster': 'cell_types'
            })
            sub_adatas.append(adata)
        adata = sc.concat(sub_adatas)
        adata.obs.reset_index(inplace=True)
        adata = adata[adata.obs.dropna().index]
        
        return adata


class BoneAtlas(SCAtlas):
    
    def __init__(self):
        self.name = 'bone'
        self.type_map = {
            'C': 'Cycling',
            'Chondrocytes': 'Chondrocyte',
            'EC-Arteriar': 'Endothelial Arteriar',
            'EC-Arteriolar': 'Endothelial Arterioral',
            'EC-Sinusoidal': 'Endothelial Sinusoidal',
            'Fibroblasts': 'Fibroblast',
            'MSPC-Adipo': 'MSPC-Adipo',
            'MSPC-Osteo': 'MSPC-Osteo',
            'Myofibroblasts': 'Myofibroblasts',
            'Osteo': 'Osteocyte',
            'Osteoblasts': 'Osteoblast',
            'Pericytes': 'Pericyte',
            'Schwann-cells': 'Schwann',
            'Smooth-muscle': 'Smooth muscle cell',
        }
        
        self.coarse_map = {
            'Chondrocyte': 'Chondrocytes',
            'Cycling': 'NA',
            'Endothelial Arteriar': 'Endothelial cells',
            'Endothelial Arterioral': 'Endothelial cells',
            'Endothelial Sinusoidal': 'Endothelial cells',
            'Fibroblast': 'Fibroblasts',
            'MSPC-Adipo': 'Mesenchymal Stem cells',
            'MSPC-Osteo': 'Mesenchymal Stem cells',
            'Myofibroblasts': 'Myofibroblasts',
            'Osteoblast': 'Osteoblasts',
            'Osteocyte': 'Osteocytes',
            'Pericyte': 'Pericytes',
            'Schwann': 'Schwann cells',
            'Smooth muscle cell': 'Smooth muscle cells'
        }


        
    def process_imp(self):
        import scanpy as sc
        meta = pd.read_csv('atlas/bone/metadata.csv')
        #counts = pd.read_csv('atlas/bone/counts.normalized.csv', index_col=0).transpose()
        counts = pd.read_parquet('atlas/bone/counts.normalized.parquet').transpose()
        
        meta.index = meta['NAME'].values
        
        merged = counts.merge(meta, right_on='NAME', left_index=True, how='inner')[['Harmonized Label']]
        
        merged = merged.rename(columns={'Harmonized Label': 'cell_types'})
    
        adata = sc.AnnData(counts, obs=merged)
    
        return adata


class GBMAtlas(SCAtlas):
    name = 'GBM2'
    
    def __init__(self):
        self.name = 'GBM2'
    
    def process_imp(self):
        import scanpy as sc
        self.type_map = {
            'BCells': 'B-cell',
            'Endo': 'Endothelial',
            'Glioma': 'Glioma',
            'Myeloid': 'Myeloid',
            'Oligo': 'Oligodendrocyte',
            'Other': 'NA',
            'Pericytes': 'Pericyte',
            'TCells': 'T-cell'
        }
        
        self.coarse_map = {
            'B-cell': 'Lymphocytes',
            'Endothelial': 'Endothelial cells',
            'Glioma': 'Glioma cells',
            'Myeloid': 'Myeloid cells',
            'Oligodendrocyte': 'Oligodendrocytes',
            'Other': 'NA',
            'Pericyte': 'Pericytes',
            'T-cell': 'Lymphocytes'
        }

        
        adata = sc.read_h5ad('atlas/GBM2/adata.h5')
        meta = pd.read_csv('/mnt/ssd/paul/tissue-seg/atlas/GBM2/Meta_GBM.txt', index_col=0)[['Assignment', 'SubAssignment']]
        meta = meta.iloc[1:]
        adata.obs = meta.rename(columns={
            'Assignment': 'cell_types',
            'SubAssignment': 'cell_types_lvl2'
        })
        
        assert np.array_equal(pd.read_csv('/mnt/ssd/paul/tissue-seg/atlas/GBM2/Meta_GBM.txt')['NAME'].values[1:], adata.obs_names.values)
        return adata
    


class HCC2Atlas(SCAtlas):
    def __init__(self):
        self.name = 'HCC2'
        
        self.coarse_map = {
            'B cells': 'Lymphocytes',
            'CD4-CD69-memory T cells': 'Lymphocytes',
            'CD4-FOXP3-regulatory T cells': 'Lymphocytes',
            'CD4-IL7R-central memory T cells': 'Lymphocytes',
            'CD4-KLRB1-T cells': 'Lymphocytes',
            'CD8-CD69-memory T cells': 'Lymphocytes',
            'CD8-GZMH-effector T cells': 'Lymphocytes',
            'CD8-GZMK-effector memory T cells': 'Lymphocytes',
            'Cholangiocytes': 'Cholangiocytes',
            'Hepatocytes': 'Hepatocytes',
            'MAIT': 'Lymphocytes',
            'NK-CD160-tissue resident': 'Lymphocytes',
            'NK-GNLY-circulatory': 'Lymphocytes',
            'Plasma cells': 'Plasma cells',
            'T cells-MKI67-proliferative': 'Lymphocytes',
            'c0-LUM-inflammatory CAF': 'Fibroblasts',
            'c0-S100A8-Monocyte': 'Monocytes',
            'c0-VWF-endothelial': 'Endothelial cells',
            'c1-ANGPT2-endothelial': 'Endothelial cells',
            'c1-CXCL10-M1 Macrophage': 'Macrophages',
            'c1-MYH11-vascular CAF': 'Fibroblasts',
            'c2-APOA1-hepatocyte like CAF': 'Fibroblasts',
            'c2-CCL4L2-M2 Macrophage': 'Macrophages',
            'c2-CRHBP-endothelial': 'Endothelial cells',
            'c3-CCL5-endothelial': 'Endothelial cells',
            'c3-TPSB2-Mast cells': 'Mast cells',
            'c4-RGS5-endothelial': 'Endothelial cells',
            'tumor': 'Cancer epithelial cells',
            'unspecified': 'NA'
        }
    
    def process_imp(self):
        import scanpy as sc
        df = pd.read_parquet('atlas/HCC2/counts.parquet')
        meta = pd.read_csv('atlas/HCC2/GSE229772_cell_subtypes.txt', sep='\t')
        
        adata = sc.AnnData(df)
        meta['sample'] = [s.replace('-', '.') for s in meta['sample'].values]
        merged = adata.obs.merge(meta, left_index=True, right_on='sample', how='left')[['subtype']]
        merged = merged.rename(columns={
            'subtype': 'cell_types'
        })
        
        adata.obs = merged
        
        adata = adata[~adata.obs['cell_types'].isna()]
        return adata


class AMLAtlas(SCAtlas):

    def __init__(self):
        self.name = 'AML'
    
    def process_imp(self):
        df = pd.read_csv('/mnt/ssd/paul/tissue-seg/atlas/AML/stroma.leuk.ctrl.TP4K.txt', sep='\t')
        meta = pd.read_csv('/mnt/ssd/paul/tissue-seg/atlas/AML/stroma.leuk.ctrl.meta.txt', sep='\t')
        raise ValueError('TODO')
    
    
class HCCAtlas(SCAtlas):
    def __init__(self):
        self.name = 'AML'
    
    def process_imp(self):
        import scanpy as sc
        meta = pd.read_csv('atlas/HCC/GSE149614_HCC.metadata.updated.txt', index_col=0, sep='\t')
        adata = sc.read_h5ad('atlas/HCC/adata.h5ad')
        merged = meta.merge(adata.obs, left_index=True, right_index=True, how='inner')
        adata = adata[merged.index]
        gene_list = []
        for samp in tqdm(np.unique(merged['sample'])):
            sub = merged[merged['sample'] == samp]
            sub_adata = adata[sub.index]
            df = sub_adata.to_df()
            non_nan = df.columns[df.notna().all()].tolist()
            gene_list.append(non_nan)
        #adata.obs = merged
        
        intersection_array = np.array(gene_list[0])
        for sublist in gene_list[1:]:
            intersection_array = np.intersect1d(intersection_array, sublist)

        # Convert the numpy array back to a list if needed
        intersection_list = intersection_array.tolist()
        adata = adata[:, intersection_list]
        adata.obs = merged[['celltype']].rename(columns={'celltype': 'cell_types'})
        
        #a = pd.read_csv('/mnt/ssd/paul/ST-histology-loader/atlas/HCC/test_ab', sep='\t', index_col=0)
        #counts = pd.read_csv('atlas/HCC/GSE149614_HCC.scRNAseq.S71915.count.txt', sep='\t')
        return adata
    
    
class MELAtlas(SCAtlas):
    
    def __init__(self):
        self.name = 'MEL'
        self.type_map = {
            'B.cell': 'B-cell',
            'CAF': 'CAF',
            'Endo.': 'Endothelial',
            'Macrophage': 'Macrophage',
            'Mal': 'Cancer',
            'NK': 'NK-cell',
            'T.CD4': 'T-cell',
            'T.CD8': 'T-cell',
            'T.cell': 'T-cell'  
        }
    
    def process_imp(self):
        import scanpy as sc
        meta = pd.read_csv('atlas/MEL/GSE115978_tpm.csv', index_col=0).transpose()
        #meta2 = pd.read_csv('atlas/skin/GSE115978_counts.csv')
        meta3 = pd.read_csv('atlas/MEL/GSE115978_cell.annotations.csv', index_col=0)
        
        adata = sc.AnnData(meta)
        adata.obs['cell_types'] = meta3.loc[adata.obs.index]['cell.types'].values
        adata = adata[adata.obs.index[adata.obs['cell_types'] != '?']]
        
        return adata
        
        
class BreastCancerAtlas(SCAtlas):
    
    name = 'BRAC'
    
    def process_imp(self):
        import scanpy as sc
        adata = sc.read_10x_mtx('/mnt/ssd/paul/tissue-seg/atlas/BRAC/mex')
        meta = pd.read_csv("/mnt/ssd/paul/tissue-seg/atlas/BRAC/Whole_miniatlas_meta.csv", index_col=0)
        meta = meta.iloc[1:]
        obs = meta.loc[adata.to_df().index]
        adata.obs = obs[['celltype_major', 'celltype_minor', 'celltype_subset', 'subtype']]
        adata.obs = adata.obs.rename(columns={
            'celltype_major': 'cell_types',
            'celltype_minor': 'cell_types_lvl2',
            'celltype_subset': 'cell_types_lvl3'
        })
        return adata

class BreastCancer3Atlas(SCAtlas):
    def __init__(self):
        self.name = 'breast3'
    
    def process_imp(self):
        import scanpy as sc
        adata =  sc.read_h5ad('atlas/breast3/sub.h5ad')
        from scanpy.queries import biomart_annotations

        # Query BioMart for annotations
        annotations = biomart_annotations(
            'hsapiens',
            ['ensembl_gene_id', 'external_gene_name'],
            use_cache=True
        )
        
        annotations.index = annotations['ensembl_gene_id']
        df = pd.DataFrame(adata.var_names).merge(annotations, left_on=0, right_index=True, how='left')
        adata.var_names = df['external_gene_name'].fillna(df[0]).values
        return adata


class HGSOCCancerAtlas(SCAtlas):
    
    def __init__(self):
        self.name = 'HGSOC'
        self.type_map = {
            'B cell': 'B-cell',
            'Ciliated': 'Ciliated',
            'Empty/Epithelial cell': 'Epithelial',
            'Fibroblast': 'Fibroblast',
            'Lymphocytes': 'Lymphocytes',
            'Macrophage': 'Macrophage',
            'Macrophages': 'Macrophage',
            'NK cell': 'NK-cell',
            'Smooth muscle cells': 'Smooth muscle cell',
            'Stromal fibroblasts': 'Fibroblast',
            'T cell': 'T-cells',
            'Unciliated epithelia 1': 'Epithelial',
            'Unciliated epithelia 2': 'Epithelial',
            'Endothelia': 'Endothelial',
            'Endothelial cell': 'Endothelial',
            'Epithelial cell': 'Epithelial'
        }
        
        self.coarse_map = {
            # cancer cells missing?????
            'B-cell': 'Lymphocytes',
            'Ciliated': 'Ciliated cells',
            'Endothelial': 'Endothelial cells',
            'Epithelial': '',
            'Fibroblast': '',
            'Lymphocytes': '',
            'Macrophage': '',
            'NK-cell': '',
            'Smooth muscle cell': '',
            'T-cells': ''
        }
    
    def process_imp(self):
        import scanpy as sc
        patient_map = dict(zip(
            ['GSM' + str(i) for i in range(5276933, 5276955)], 
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ))
        
        adatas = []
        for mex in os.listdir('atlas/HGSOC/regner'):
            if not os.path.isdir(os.path.join('atlas/HGSOC/regner', mex)):
                continue
            adata = sc.read_10x_mtx(os.path.join('atlas/HGSOC/regner', mex))
            suffix = '_' + str(patient_map[mex])
            adata.obs.index = [i + suffix for i in adata.obs.index]
            adatas.append(adata)
        
        adata = sc.concat(adatas)
        meta = pd.read_csv('atlas/HGSOC/regner/hgsoc_meta.csv', index_col=0)
        meta = meta.rename(columns={'SingleR': 'cell_types'})
        meta = meta[['cell_types']]
        meta = meta.dropna()
        
        merged = meta.merge(adata.obs, left_index=True, right_index=True)
        #merged.index = merged['Cell']
        adata = adata[merged.index]
        adata.obs = merged
        
        return adata


class RCCAtlas(SCAtlas):
    def __init__(self):
        self.name = 'RCC'
        self.type_map = {
            '41BB-Hi CD8+ T cell': 'T-cells',
            '41BB-Lo CD8+ T cell': 'T-cells',
            'B cell': 'B-cells',
            'CD16+ Monocyte': 'Monocytes',
            'CD16- Monocyte': 'Monocytes',
            'CD1C+ DC': 'Dentritic cells',
            'CLEC9A+ DC': 'Dentritic cells',
            'CXCL10-Hi TAM': 'TAMs',
            'Cycling CD8+ T cell': 'T-cells',
            'Cycling TAM': 'TAMs',
            'Cycling Tumor': 'Tumor',
            'Effector T-Helper': 'Helper T-cells',
            'Endothelial': 'Endothelial cells',
            'FGFBP2+ NK': 'NK-cells',
            'FGFBP2- NK': 'NK-cells',
            'FOLR2-Hi TAM': 'TAMs',
            'Fibroblast': 'Fibroblasts',
            'GPNMB-Hi TAM': 'TAMs',
            'LowLibSize Macrophage': 'Macrophages',
            'MX1-Hi CD8+ T cell': 'T-cells',
            'Mast cell': 'Mast cells',
            'Memory T-Helper': 'Helper T-cells',
            'Misc/Undetermined': 'NA',
            'MitoHigh CD8+ T cell': 'T-cells',
            'MitoHigh Myeloid': 'Myeloid cells',
            'MitoHigh NK': 'NK-cells',
            'MitoHigh T-Helper': 'Helper T-cells',
            'NKT': 'NKT-cells',
            'Plasma cell': 'Plasma cells',
            'T-Reg': 'Treg-cells',
            'TP1': 'Tumor cells',
            'TP2': 'Tumor cells',
            'VSIR-Hi TAM': 'TAMs'
        }

    
    def process_imp(self):
        import scanpy as sc
        barcodes = pd.read_csv('atlas/RCC/barcodes.tsv', sep='\t', header=None).values.flatten()
        genes = pd.read_csv('atlas/RCC/genes.tsv', sep='\t', header=None)[0].values.flatten()
        adata = sc.read_mtx('atlas/RCC/matrix.mtx').transpose()
        adata.obs_names = barcodes
        adata.var_names = genes
        
        meta = pd.read_csv('atlas/RCC/Final_SCP_ClusterFile.txt', sep='\t')
        meta = meta.iloc[1:]
        merged = adata.obs.merge(meta, left_index=True, right_on='NAME', how='inner')
        merged = merged.rename(columns={'FinalCellType': 'cell_types'})[['cell_types']]
        adata = adata[merged.index]
        adata.obs = merged
        return adata
        

class ColonCancerAtlas(SCAtlas):
    def __init__(self):
        self.name = 'colon'
        
        self.coarse_map = {
            'B-cell': 'Lymphocytes',
            'Dentric': 'Dendritic cells',
            'Endothelial': 'Endothelial cells',
            'Epithelial Cancer': 'Cancer epithelial cells',
            'Epithelial Normal': 'Normal epithelial cells',
            'Fibroblasts': 'Fibroblasts',
            'Granulocyte': 'Granulocytes',
            'Innate Lymphoid': 'Lymphocytes',
            'Macrophage': 'Macrophages',
            'Mast': 'Mast cells',
            'Monocyte': 'Monocytes',
            'NK-cell': 'Lymphocytes',
            'Pericyte': 'Pericytes',
            'Plasma': 'Plasma cells',
            'Schwann': 'Schwann cells',
            'Smooth Muscle': 'Smooth muscle cells',
            'T-cell CD4': 'Lymphocytes',
            'T-cell CD8': 'Lymphocytes',
            'TZBTB16': 'Lymphocytes',
            'Tgd-cell': 'Lymphocytes'
        }
    
    def process_imp(self):
        import scanpy as sc
        
        self.type_map = {
            'B': 'B-cell',
            'DC': 'Dentric',
            'Endo': 'Endothelial',
            'Epi': 'Epithelial Normal',
            'EpiT': 'Epithelial Cancer',
            'Fibro': 'Fibroblasts',
            'Granulo': 'Granulocyte',
            'ILC': 'Innate Lymphoid',
            'Macro': 'Macrophage',
            'Mast': 'Mast',
            'Mono': 'Monocyte',
            'NK': 'NK-cell',
            'Peri': 'Pericyte',
            'Plasma': 'Plasma',
            'Schwann': 'Schwann',
            'SmoothMuscle': 'Smooth muscle cell',
            'TCD4': 'T-cell CD4',
            'TCD8': 'T-cell CD8',
            'TZBTB16': 'TZBTB16',
            'Tgd': 'Tgd-cell'
        }

    
        #adata = sc.read_mtx('atlas/colon/mex/matrix.mtx').transpose()
        adata = sc.read_h5ad('atlas/colon/colon.h5')
        
        meta = pd.read_csv('atlas/colon/crc10x_tSNE_cl_global.tsv', sep='\t').loc[1:]
        adata.obs.index = [i[0] for i in adata.obs.index]
        adata.obs = meta[['ClusterFull', 'ClusterMidway']]
        adata.obs = adata.obs.rename(columns={
            'ClusterFull': 'cell_types_lvl2',
            'ClusterMidway': 'cell_types',
        })
        return adata


class PAADAtlas(SCAtlas):
    def __init__(self):
        self.name = 'PAAD'
        self.type_map = {
            'B_Cells': 'B-cells',
            'DC': 'Dentritic cells',
            'Endothelial': 'Endothelial cells',
            'Hepatocyte': 'Hepatocytes',
            'Macrophage': 'Macrophages',
            'Mesenchymal': 'Mesenchymal cells',
            'Plasma_cell': 'Plasma cells',
            'T_NK': 'NKT-cells',
            'T_Regs': 'Treg-cells',
            'Tumor': 'Tumor cells',
            'XCR1_DC': 'Dentritic cells',
            'pDC_cell': 'Plasmacytoid dendritic cells'
        }
        
        self.coarse_map = {
            'B-cells': 'Lymphocytes',
            'Dentritic cells': 'Dendritic cells',
            'Endothelial cells': 'Endothelial cells',
            'Hepatocytes': 'Hepatocytes',
            'Macrophages': 'Macrophages',
            'Mesenchymal cells': 'Mesenchymal cells',
            'NKT-cells': 'Lymphocytes',
            'Plasma cells': 'Plasma cells',
            'Plasmacytoid dendritic cells': 'Lymphocytes',
            'Treg-cells': 'Lymphocytes',
            'Tumor cells': 'Cancer epithelial cells'
        }
        
    
    def process_imp(self):
        import scanpy as sc

        #raw_concat = pd.read_csv('atlas/PAAD/Biopsy_RawDGE_23042cells.csv', index_col=0).transpose()
        raw_concat = pd.read_parquet('atlas/PAAD/Biopsy_RawDGE_23042cells.parquet')
        
        meta = pd.read_csv('atlas/PAAD/complete_MetaData_70170cells_scp.csv', index_col=0)
        
        meta = meta.iloc[1:]
        
        merged = meta.merge(raw_concat, left_index=True, right_index=True, how='inner')[['Coarse_Cell_Annotations']]
        merged = merged.rename(columns={'Coarse_Cell_Annotations': 'cell_types'})
        raw_concat = raw_concat.loc[merged.index]
        
        adata = sc.AnnData(raw_concat, obs=merged)

        return adata
    
    
class TICPancancerAtlas(SCAtlas):
    def __init__(self):
            self.name = 'pancancer'
    
    def process_imp(self):
        meta_df = pd.read_csv('/mnt/ssd/paul/tissue-seg/atlas/pancancer/TICAtlas_metadata.csv')
        cell_types = meta_df['lv2_annot'].values
        print('reading integrated matrix...')
        matrix = pd.read_csv('/mnt/ssd/paul/tissue-seg/atlas/pancancer/TICAtlas_integrated_matrix.csv')
        raise ValueError('TODO')
    

class NSCLCAtlas(SCAtlas):
    def __init__(self):
        self.name = 'lung'
        self.type_map = {
            'B cells': 'B-cell',
            'Cancer cells': 'Epithelial Cancer',
            'Ciliated cells': 'Ciliated',
            'Endothelial cells': 'Endothelial',
            'Fibroblasts': 'Fibroblast',
            'Mast cells': 'Mast',
            'Myeloid cells': 'Myeloid',
            'Plasma cells': 'Plasma',
            'T cells': 'T-cell'
        }
        
        self.coarse_map = {
            'Alveolar Macrophages': 'Macrophages',
            'Alveolar cells': 'Alveolar cells',
            'CAF': 'Fibroblasts',
            'CD4+ Treg': 'Lymphocytes',
            'CD8+ Effector memory T cells': 'Lymphocytes',
            'CDKN2A Cancer cells': 'Cancer epithelial cells',
            'CXCL1 Cancer cells': 'Cancer epithelial cells',
            'Ciliated cells': 'Ciliated cells',
            'Endothelial cells': 'Endothelial cells',
            'LAMC2 Cancer cells': 'Cancer epithelial cells',
            'Lipid-associated moMac': 'Macrophages',
            'Low quality Mac': 'Macrophages',
            'Mast cells': 'Mast cells',
            'Mature naive B cells': 'Lymphocytes',
            'Monocytes': 'Monocytes',
            'NK cells': 'Lymphocytes',
            'Naive T cells': 'Lymphocytes',
            'Neutrophils': 'Neutrophils',
            'Patological Alveolar cells': 'Alveolar cells',
            'Plasma cells': 'Plasma cells',
            'Proliferating Cancer cells': 'Cancer epithelial cells',
            'Proliferating Macrophages': 'Macrophages',
            'Proliferating T/NK cells': 'Lymphocytes',
            'SOX2 Cancer cells': 'Cancer epithelial cells',
            'Smooth muscle cells': 'Smooth muscle cells',
            'cDC2/moDCs': 'Dendritic cells',
            'pDCs': 'Lymphocytes'
        }


    
    def process_imp(self):
        import scanpy as sc
        adata = sc.read_h5ad('atlas/lung/lung_sc.h5ad')
        adata.obs = adata.obs[['predicted.celltypel1', 'predicted.celltypel2']]
        adata.obs = adata.obs.rename(columns={
            'predicted.celltypel1': 'cell_types',
            'predicted.celltypel2': 'cell_types_lvl2'
        })
        adata = adata[adata.obs['cell_types'] != 'NA']
        adata.obs = adata.obs.reset_index()[['cell_types', 'cell_types_lvl2']]
        adata.__dict__['_raw'].__dict__['_var'] = adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})

        return adata


def get_cells_with_clusters(bc_matrix_path, cluster_path=None, k=None) -> sc.AnnData:
    import scanpy as sc
    cells = sc.read_10x_h5(bc_matrix_path)
    
    cells_df = cells.to_df()
    if cluster_path is None:
        from sklearn.cluster import KMeans
        print('Computing kmeans...')
        k_means = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(cells_df.values)
        joined_df = pd.Series(k_means.labels_, index=cells_df.index)
    else:
        df = pd.read_csv('/mnt/sdb1/paul/data/samples/xenium/Preview Data: FFPE Human Lung Cancer with Xenium Multimodal Cell Segmentation/analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv')
        df.index = df['Barcode']
    
        joined_df = cells_df.merge(df, left_index=True, right_index=True, how='left')['Cluster']

    assert np.array_equal(joined_df.index, cells.obs.index)
    cells.obs['cluster'] = joined_df
    
    return cells


def sc_atlas_factory(tissue) -> SCAtlas:
    
    organ_atlas_map = {
        'Breast': IDCAtlas,
        'Lung': NSCLCAtlas,
        'Ovary': HGSOCCancerAtlas,
        'Pancreas': PAADAtlas,
        'Bowel': ColonCancerAtlas,
        'Intestine': ColonCancerAtlas,
        'Colon': ColonCancerAtlas,
        'Brain': GBMAtlas,
        'Kidney': RCCAtlas,
        'Skin': MELAtlas,
        'Bone': BoneAtlas,
        'Bone marrow': BoneAtlas,
        'Femur bone': BoneAtlas,
        'Tonsil': TonsilAtlas,
        'Heart': HeartAtlas,
    }
    
    return organ_atlas_map[tissue]()


def get_atlas_from_name(name):
    for cls in SCAtlas.__subclasses__():
        if cls.name == name:
            return cls
    raise ValueError(f'No atlas with name {name}')