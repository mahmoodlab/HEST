import geopandas as gpd
import numpy as np
import pandas as pd

from hest.HESTData import HESTData


def evaluate_stromal_batch_effect(st: HESTData, stromal_threshold = 0.7, min_cells_treshold = 2):
    adata = st.adata
    
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
    