from __future__ import annotations

import os
import warnings
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from hestcore.segmentation import get_path_relative
from loguru import logger

from hest.HESTData import HESTData, unify_gene_names
from hest.utils import check_arg, verify_paths


def get_housekeeping_stromal_adata(
    st: HESTData, 
    species: str, 
    stromal_threshold = 0.7, 
    min_cells_treshold = 2, 
    unify_genes=True, 
    strict_genes=True,
    min_stromal_spot=5,
    plot_path='',
    whole_tissue=False,
):
    """ Filter the st.adata of a HEST sample to only keep stromal regions and housekeeping genes
    
    Stromal regions are determined by computing the proportion of cells classified as stromal by CellViT. If the propotion
    is greater than `stromal_threshold` the region will be determined as stromal. The st.adata genes are then filtered to only keep housekeeping genes
    
    """
    
    adata = unify_gene_names(st.adata, 'hsapiens') if unify_genes else st.adata.copy()
    
    check_arg(species.lower(), 'species', ['human', 'mouse'])
    
    asset_dir = get_path_relative(__file__, '../../assets/')
    
    filename = 'MostStable_Human.csv' if species.lower() == 'human' else 'MostStable_Mouse.csv'
    path_genes = os.path.join(asset_dir, filename)
    
    housekeep_genes = pd.read_csv(path_genes, sep=';')['Gene name'].values
    missing_genes = np.setdiff1d(housekeep_genes, adata.var_names)
    if len(missing_genes) > 0:
        if strict_genes:
            raise ValueError(f"The following housekeeping genes are missing in st.adata: {missing_genes}. If you still want to evaluate the batch effect, pass strict_genes=False")
        else:
            warnings.warn(f"The following housekeeping genes are missing in st.adata: {missing_genes}")
    common_genes = np.intersect1d(housekeep_genes, adata.var_names)
    
    if not whole_tissue:
        logger.info("Detecting stromal regions under tissue from CellViT segmentation...")
        cellvit_cells = st.get_shapes('cellvit', 'he').shapes
        cell_centers = cellvit_cells.copy()
        cell_centers['geometry'] = cellvit_cells.centroid
        
        xy = adata.obsm['spatial']
        spot_centers = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), index=adata.obs.index)
        
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
        
        union_contours = st.tissue_contours.union_all()
        
        strom_stromal_spots_under_tissue = stromal_spots[stromal_spots.geometry.within(union_contours)]
        
        adata = adata[strom_stromal_spots_under_tissue.index]
        
        logger.info(f"Detected {len(adata)} stromal regions under tissue")
        
        if len(adata) < min_stromal_spot:
            raise Exception(f"Detected less than {min_stromal_spot} stromal patches under tissue")
    
    adata = adata[:, common_genes]


    st.adata = adata
    
    if plot_path is not None:
        st.save_spatial_plot(plot_path)
    return adata


def plot_umap(adata_list: List[sc.AnnData], plot_path, names):
    import matplotlib.pyplot as plt
    import scanpy as sc
    import umap
    
    common_genes = adata_list[0].var_names
    for adata in adata_list[1:]:
        if not np.array_equal(adata.var_names, common_genes):
            raise ValueError("Each adata in adata_list must have the same var_names")
    

    concat_adata = sc.concat(adata_list)
    arr = concat_adata.to_df().values
    labels = np.zeros(len(concat_adata))
    start = 0
    for i, adata in enumerate(adata_list):
        l = adata.shape[0]
        labels[start:start+ l] = i
        start = start + l
    
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', verbose=1)
    embedding = umap_model.fit_transform(arr)
    
    plt.figure(figsize=(10, 8))
    for i, arr in enumerate(adata_list):
        name = names[i]
        plt.scatter(embedding[labels == i, 0], embedding[labels == i, 1], label=name, s=0.8)
    plt.title('UMAP')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(scatterpoints=1, markerscale=4)
    plt.savefig(plot_path)