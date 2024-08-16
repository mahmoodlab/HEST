import os
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd

from hest.HESTData import HESTData, unify_gene_names
from hest.utils import check_arg, verify_paths
from hestcore.segmentation import get_path_relative


def evaluate_stromal_batch_effect(st: HESTData, species: str, stromal_threshold = 0.7, min_cells_treshold = 2, unify_genes=True, strict_genes=True):
    adata = unify_gene_names(st.adata, 'hsapiens') if unify_genes else st.adata
    
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
    
    st.adata = st.adata[strom_stromal_spots_under_tissue.index]
    
    st.save_spatial_plot('.')
    
    a = 1
    