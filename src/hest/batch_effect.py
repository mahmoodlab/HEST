from __future__ import annotations

import os
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from hestcore.segmentation import get_path_relative
from loguru import logger
from tqdm import tqdm

from hest.HESTData import HESTData, load_hest, unify_gene_names
from hest.utils import check_arg


def filter_housekeeping(adata, species='human'):
    adata = adata.copy()
    check_arg(species.lower(), 'species', ['human', 'mouse'])
    asset_dir = get_path_relative(__file__, '../../assets/')
    
    filename = 'MostStable_Human.csv' if species.lower() == 'human' else 'MostStable_Mouse.csv'
    path_genes = os.path.join(asset_dir, filename)
    
    housekeep_genes = pd.read_csv(path_genes, sep=';')['Gene name'].values
    missing_genes = np.setdiff1d(housekeep_genes, adata.var_names)
    if len(missing_genes) > 0:
        raise ValueError(f"The following housekeeping genes are missing in st.adata: {missing_genes}. Make sure {missing_genes} or one of its aliases is in st.adata.var_names")
    common_genes = np.intersect1d(housekeep_genes, adata.var_names)
    adata = adata[:, common_genes]
    return adata


def filter_stromal_housekeeping(
    st: HESTData,
    species: str, 
    stromal_threshold = 0.7, 
    min_cells_treshold = 2, 
    unify_genes=False, 
    min_stromal_spot=5,
    plot_dir=None,
    name='',
    whole_tissue=False,
    verbose=False
):
    """ Filter the st.adata of a HEST sample to only keep stromal regions and housekeeping genes
    
        Stromal regions are determined by computing the proportion of cells classified as stromal by CellViT. If the propotion
        is greater than `stromal_threshold` the region will be determined as stromal. The st.adata genes are then filtered to only keep housekeeping genes

        Args:
            st (HESTData): HEST sample to filter
            species (str): species (can be 'human' or 'mouse')
            stromal_threshold (float, optional): If the propotion
                of stromal cells within a spot is greater than `stromal_threshold` the region will be determined as stromal. Defaults to 0.7.
            min_cells_treshold (int, optional): spots containing less than `min_cells_treshold` will be filtered out. Defaults to 2.
            unify_genes (bool, optional): whenever to maps gene aliases to their parents. Defaults to False.
            min_stromal_spot (int, optional): minimum number of stromal_pots detected before throwing an exception. Defaults to 5.
            plot_dir (str, optional): if not None, will save a plot of the filtered adata in that directory. Defaults to None.
            name (str, optional): name. Defaults to ''.
            whole_tissue (bool, optional): whenever to keep the whole tissue or only stromal regions. Defaults to False.
    """
    
    adata = unify_gene_names(st.adata, 'human') if unify_genes else st.adata.copy()
    
    adata = filter_housekeeping(adata, species)
    
    xy = adata.obsm['spatial']
    spot_centers = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), index=adata.obs.index)
    
    simplified_contours = st.tissue_contours.simplify(10)
    union_contours = simplified_contours.union_all()
    
    under_tissue_mask = spot_centers.geometry.within(union_contours)
    adata = adata[under_tissue_mask]
    spot_centers = spot_centers[under_tissue_mask]
    
    if not whole_tissue:
        if verbose:
            logger.info("Detecting stromal regions under tissue from CellViT segmentation...")
        cellvit_cells = st.get_shapes('cellvit', 'he').shapes
        cell_centers = cellvit_cells.copy()
        cell_centers['geometry'] = cellvit_cells.centroid
        
        nearest_spot_idx = spot_centers.geometry.sindex.nearest(cell_centers.geometry)[1]
        cell_centers['spot_idx'] = nearest_spot_idx
        
        spot_counts = pd.DataFrame(np.zeros((len(spot_centers), 2)), columns=['stromal', 'non-stromal'], dtype=int, index=spot_centers.index)
        
        
        stromal_cells = cell_centers[cell_centers['class'] == 'Connective']
        non_stromal_cells = cell_centers[~(cell_centers['class'] == 'Connective')]
        
        spot_counts = spot_counts.reset_index()
        stromal_count = stromal_cells['spot_idx'].value_counts()
        spot_counts['stromal'] = spot_counts['stromal'].add(stromal_count, fill_value=0)
        
        non_stromal_count = non_stromal_cells['spot_idx'].value_counts()
        spot_counts['non-stromal'] = spot_counts['non-stromal'].add(non_stromal_count, fill_value=0)
        spot_counts.index = adata.obs.index


        spot_counts['stromal_pct'] = spot_counts['stromal'] / (spot_counts['stromal'] + spot_counts['non-stromal'])
        spot_counts.fillna(0)
        
        assert spot_counts[['stromal', 'non-stromal']].sum().sum() == len(cellvit_cells)
        
        stromal_spots = spot_centers[(spot_counts['stromal_pct'] > stromal_threshold) & (spot_counts['stromal'] > min_cells_treshold)]
        
        adata = adata[stromal_spots.index]
        
        if verbose:
            logger.info(f"Detected {len(adata)} stromal regions")
        
        if len(adata) < min_stromal_spot:
            raise Exception(f"Detected less than {min_stromal_spot} stromal patches")

    st.adata = adata
    
    if plot_dir is not None:
        st.save_spatial_plot(plot_dir, name)
    return adata


def filter_hest_stromal_housekeeping(meta_df: pd.DataFrame, hest_dir, whole_tissue=False, unify_genes=False, verbose=False) -> List[HESTData]:
    """ Filter the genes of HESTData samples, such that:
    - only stable housekeeping genes are kept (see assets/MostStable_{species}.csv).
    - only stromal regions are kept (determined from CellViT segmentation)
    
    The lists of most stable housekeeping genes across organs were taken from https://housekeeping.unicamp.br/?download

    Args:
        meta_df (pd.DataFrame): panda dataframe containing the following columns ['id', 'species']
        whole_tissue (bool, optional): whenever to only keep stromal regions. Defaults to False.
        unify_genes (bool, optional): whenever to all the gene names beforehand. Defaults to False.

    Returns:
        List[HESTData]: filtered list of sc.AnnData containing only stromal regions and stable housekeeping genes
    """
    adata_list = []
    for _, row in tqdm(meta_df.iterrows()):
        id = row['id']
        species = 'human' if row['species'] == 'Homo sapiens' else 'mouse'
        logger.debug(f'Filtering {id}')
        st = load_hest(hest_dir, [id])[0]
        adata = filter_stromal_housekeeping(
            st, 
            species, 
            whole_tissue=whole_tissue,
            name=st.meta['id'],
            unify_genes=unify_genes,
            verbose=verbose
        )     
        adata_list.append(adata)
    return adata_list


def _concat_adata_and_labels(adata_list, labels):
    import scanpy as sc
    concat_adata = sc.concat(adata_list)
    concat_arr = concat_adata.to_df().values
    concat_labels = ['' for _ in range(len(concat_adata))]
    start = 0
    for i, adata in enumerate(adata_list):
        l = adata.shape[0]
        concat_labels[start:start+ l] = [labels[i]] * l
        start = start + l
    return concat_arr, concat_labels


def get_silhouette_score(adata_list: List[sc.AnnData], labels, random_state=42) -> float:
    """ Compute the silhouette score for `adata_list`, cluster memberships are passed in `labels` (len(labels) == len(adata_list)) """
    from sklearn.metrics import silhouette_score
    
    if len(adata_list) == 0:
        raise ValueError("adata_list can't be empty")
    
    if len(adata_list) != len(labels):
        raise ValueError('adata_list and labels must be the same length')
    
    concat_arr, concat_labels = _concat_adata_and_labels(adata_list, labels)
    
    s_score = silhouette_score(concat_arr, concat_labels, random_state=random_state)
    return s_score

def plot_umap(
    adata_list: List[sc.AnnData], 
    labels, plot_path, 
    random_state=42, 
    umap_kwargs={}, 
    verbose=False
):
    """ Create UMAP plot (n=2) for `adata_list`, cluster memberships are passed in `labels` (len(labels) == len(adata_list)) """
    import matplotlib.pyplot as plt
    import umap
    
    if len(adata_list) == 0:
        raise ValueError("adata_list can't be empty")
    
    if len(adata_list) != len(labels):
        raise ValueError('adata_list and labels must be the same length')
    
    common_genes = adata_list[0].var_names
    for adata in adata_list[1:]:
        if not np.array_equal(adata.var_names, common_genes):
            raise ValueError("Each adata in adata_list must have the same var_names")
    

    concat_arr, concat_labels = _concat_adata_and_labels(adata_list, labels)
    
    umap_model = umap.UMAP(metric='euclidean', verbose=verbose, random_state=random_state, **umap_kwargs)
    embedding = umap_model.fit_transform(concat_arr)
    
    plt.figure(figsize=(10, 8))
    if len(adata_list) > 10:
        colors = plt.get_cmap('tab20').colors
    else:
        colors = plt.get_cmap('tab10').colors
        
    unique_labels = np.unique(labels)
    concat_labels = np.array(concat_labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        mask = concat_labels == label
        color = colors[i % len(colors)]
        plt.scatter(embedding[mask, 0], embedding[mask, 1], label=label, s=0.5, color=color)
    plt.title('UMAP')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(scatterpoints=1, markerscale=4)
    plt.savefig(plot_path)
    

def correct_batch_effect(adata_list: List[sc.AnnData], batch=None, method='combat', method_kwargs={}) -> List[sc.AnnData]:
    """ Apply a batch effect correction method to a list of Spatial Transcriptomics expressions

    Args:
        adata_list (List[sc.AnnData]): list of sc.AnnData containing gene expression
        batch (List[int], optional): list of integers corresponding to the batch membership of each adata in adata_list. 
        (i.e. [1, 1, 0] means that the first two adata are in the same batch and the third one is in a different batch). Defaults to None.
        method (str, optional): bacth correction method, must be in ['combat', 'mnn', 'harmony']. Defaults to 'combat'.
        method_kwargs (dict, optional): batch correction method kwargs. Defaults to {}.
        
    Returns:
        List[sc.AnnData]: batch corrected list of gene expression
    """
    import scanpy as sc
    from scanpy.pp import combat
    from scanpy.external.pp import mnn_correct, harmony_integrate
    
    from anndata import concat
    
    if batch is not None:
        if len(batch) != len(adata_list):
            raise ValueError('adata_list and batch must be the same length')
        batch_names = batch
    else:
        batch_names = np.arange(len(adata_list))
        
        
    check_arg(method, 'method', ['combat', 'mnn', 'harmony'])
    if method == 'combat':
        batch_effect_method = combat
    elif method == 'mnn':
        batch_effect_method = mnn_correct
    elif method == 'harmony':
        batch_effect_method = harmony_integrate
        
    
    for i in range(len(adata_list)):
        adata_list[i].obs['batch'] = batch_names[i]
    
    concat_adata = concat(adata_list)
    batch_effect_method(concat_adata, **method_kwargs)
    
    adata_list_res = []
    for batch_name in batch_names:
        adata_list_res.append(concat_adata[concat_adata.obs['batch'] == batch_name])
    
    return adata_list_res
    
    
    