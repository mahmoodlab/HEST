from __future__ import annotations

import os
import sys
import traceback
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from hest.path_utils import get_path_relative
from loguru import logger
from tqdm import tqdm

from hest.io.seg_readers import GeojsonCellReader
from hest.utils import get_n_threads
from hest.utils import verify_paths


def cellvit_light_error():
    import traceback
    traceback.print_exc()
    raise Exception("cellvit_light is not installed. Please install CellViT with `pip install cellvit-light`")
    
    
class HoverFastSegmenter():
    def segment_cells(self, wsi_path: str, name: str, src_pixel_size: float=None, dst_pixel_size: float=0.25, batch_size=2, gpu=0, save_dir='results/segmentation') -> gpd.GeoDataFrame:
        import hoverfast
        
        output_path = os.path.join(save_dir, name)
        
        sys.argv = [
            '',
            "infer_wsi",
            wsi_path,
            "-m", 'hoverfast_crosstissue_best_model.pth',
            "-n",
            "20",
            "-o",
            output_path
        ]
        
        hoverfast.main()
        
        
class CellSegmenter:
    
    def segment_cells(
        self, 
        wsi_path: str, 
        name: str,
        pixel_size: str,
        **kwargs
    ): 
        return self._segment_cells_imp(wsi_path, name, pixel_size, **kwargs)
    
    @abstractmethod
    def _segment_cells_imp(
        self, 
        wsi_path: str, 
        name: str,
        pixel_size,
        **kwargs
    ):
        pass
    
    
class CellViTSegmenter(CellSegmenter):
    MODELS_SRC_MAP = {
        'CellViT-256-x20.pth': '1w99U4sxDQgOSuiHMyvS_NYBiz6ozolN2',
        'CellViT-256-x40.pth': '1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q',
        'CellViT-SAM-H-x40.pth': '1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV',
        'CellViT-SAM-H-x20.pth': '1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30'
    }
    
    def _preprocess(self, wsi_path: str, name: str, src_pixel_size, dst_pixel_size, save_dir, processes=8):
        try:
            import cellvit_light
        except:
            cellvit_light_error()
        
        batch_size = 8
        if src_pixel_size is None:
            src_pixel_size = dst_pixel_size
            warnings.warn("no src_pixel_size provided, slide will not be rescaled. Provide a pixel size in um/px for using the right scale")
        
        output_path = os.path.join(save_dir, name)
        
        wsi_extension = wsi_path.split('.')[-1]
        supported_extensions = ['tif', 'svs']
        if wsi_extension not in supported_extensions:
            raise Exception(f"Unsupported format: {wsi_extension}, CellViT supports: {supported_extensions}")
        
        config = f"""
        min_intersection_ratio: 0.0
        normalize_stains: false
        output_path: {output_path}
        overwrite: true
        patch_overlap: 6.25
        patch_size: 1024
        processes: {processes}
        target_mpp: {dst_pixel_size}
        wsi_extension: {wsi_extension}
        wsi_paths: "{wsi_path}"
        wsi_properties:
            magnification: 40
            slide_mpp: {src_pixel_size}
        batch_size: {batch_size}
        """
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            config_path = os.path.join(temp_dir, name + '.yaml')
            with open(config_path, 'w') as file:
                file.write(config)
            cellvit_light.run_preprocessing(config_path)
        return output_path
    
    
    def _verify_model(self, model_path, model):
        import gdown
        
        if not os.path.exists(model_path):
            print(f'Model not found at {model_path}, downloading...')
            gdrive_id = self.MODELS_SRC_MAP[model]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdown.download(id=gdrive_id, output=model_path, quiet=False)
        else:
            print(f'Found model at {model_path}')
        
    
    def _segment_cells_imp(
        self, 
        wsi_path: str, 
        name: str, 
        pixel_size: float, 
        dst_pixel_size: float=0.5, 
        batch_size=2, 
        gpu_ids=[0], 
        save_dir='results/segmentation',
        model='CellViT-SAM-H-x20.pth'
    ) -> str:
        try:
            import cellvit_light
        except:
            print(traceback.format_exc())
            cellvit_light_error()
            
        verify_paths([wsi_path])
        
        model_dir = get_path_relative(__file__, '../../../models')
        model_path = os.path.join(model_dir, model)
        
        if model in self.MODELS_SRC_MAP:
            self._verify_model(model_path, model)
        else:
            if not os.path.exists(model_path):
                raise Exception(f"Can't find model weights {model_path}, only {list(self.MODELS_SRC_MAP.keys())} can be downloaded automatically")
        
        preprocess_path = self._preprocess(wsi_path, name, pixel_size, dst_pixel_size, save_dir)
        
        original_argv = sys.argv
        
        all_entries = os.listdir(preprocess_path)
        sub_name = [entry for entry in all_entries if os.path.isdir(os.path.join(preprocess_path, entry))][0]     

        sys.argv = [
            '',
            "--model", model_path,
            "--geojson",
            "--batch_size", str(batch_size),
            "--magnification",
            "40",
            "process_wsi",
            "--wsi_path", wsi_path,
            "--patched_slide_path", os.path.join(preprocess_path, sub_name),
        ]
        
        gpu_args = ["--gpu_ids"]
        for gpu in gpu_ids:
            gpu_args.append(str(gpu))
            
        sys.argv = sys.argv[0:1] + gpu_args + sys.argv[1:]
        
        cellvit_light.segment_cells()
        sys.argv = original_argv
        
        folder_name = [f for f in os.listdir(preprocess_path) if os.path.isdir(os.path.join(preprocess_path, f))][0]
        
        cell_seg_path = os.path.join(preprocess_path, folder_name, 'cell_detection', 'cells.geojson')
        
        return cell_seg_path


def cell_segmenter_factory(method: str) -> CellSegmenter:
    if method == 'cellvit':
        return CellViTSegmenter()
    elif method == 'hoverfast':
        return HoverFastSegmenter()
    else:
        raise ValueError(f"cell segmenter should be one of the following: ['cellvit', 'hoverfast']")


def segment_cellvit(
    wsi_path: str, 
    name: str, 
    src_pixel_size: float=None, 
    dst_pixel_size: float=0.5, 
    batch_size=2, 
    gpu_ids=[0], 
    save_dir='results/segmentation',
    model: str='CellViT-SAM-H-x20.pth'
) -> str:
    """ Segment nuclei with CellViT

    Args:
        wsi_path (str): path to slide to segment (.tiff prefered)
        name (str): name of run
        src_pixel_size (float, optional): pixel size (um/px) of the slide at wsi_path. Defaults to None.
        dst_pixel_size (float, optional): patch will be resized to this (um/px) before being fed to CellViT. Defaults to 0.25.
        batch_size (int, optional): batch_size. Defaults to 2.
        gpu_ids (List[int], optional): list of gpu ids to use during inference. Defaults to [0].
        save_dir (str, optional): directory where to save the output. Defaults to 'results/segmentation'.
        model (str, optional): name of model weights to use. Defaults to 'CellViT-SAM-H-x20.pth'. List of weights available:

            - 'CellViT-256-x20.pth' 
            - 'CellViT-256-x40.pth'
            - 'CellViT-SAM-H-x40.pth' 
            - 'CellViT-SAM-H-x20.pth'

    Returns:
        str: path to the result segmentation .geojson
    """
    segmenter = CellViTSegmenter()
    return segmenter.segment_cells(
        wsi_path, 
        name, 
        src_pixel_size, 
        dst_pixel_size=dst_pixel_size, 
        batch_size=batch_size, 
        gpu_ids=gpu_ids, 
        save_dir=save_dir,
        model=model
    )
    

def read_spots_gdf(path):
    points_df = pd.read_parquet(path)
    points_df = points_df.rename(columns={
        'pxl_col_in_fullres': 'x',
        'pxl_row_in_fullres': 'y'
    })
    from shapely.geometry import Point
    
    points_geometry = gpd.points_from_xy(points_df['x'], points_df['y'])
    points_gdf = gpd.GeoDataFrame(points_df[['barcode']], geometry=points_geometry)
    return points_gdf

    
def read_seg(cells) -> gpd.GeoDataFrame:
    if isinstance(cells, str):
        return GeojsonCellReader().read_gdf(cells)
    elif isinstance(cells, gpd.GeoDataFrame):
        return cells
    else:
        ValueError("cells must be either a path (str) or a GeoDataFrame, not ", type(cells))
        
def read_adata(adata) -> sc.AnnData: # type: ignore
    import scanpy as sc
    
    if isinstance(adata, sc.AnnData):
        return adata
    elif isinstance(adata, str):
        return sc.read_10x_h5(adata)
    else:
        ValueError("cells must be either a path (str) or a sc.AnnData, not ", type(adata))
    
    
def _sjoin(chunk, cell_gdf):
    return gpd.sjoin(chunk, cell_gdf, how='left', predicate='within')
    
def assign_spot_to_cell(cell_gdf, point_gdf, n_workers=-1):
    """ Return a spot index to cell_id assigment as a pd.Series """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    logger.info('matching spots to cells...')
    assignments = np.zeros(len(point_gdf), dtype=int)
        
    n_threads = get_n_threads(n_workers)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        chunk_size = len(point_gdf) // n_threads
        
        chunks = [point_gdf.iloc[i:i+chunk_size] for i in range(0, len(point_gdf), chunk_size)]
        
        futures = [executor.submit(_sjoin, chunk, cell_gdf) for chunk in chunks]
        i = 0
        for future in futures:
            spatial_join = future.result()
            spatial_join = spatial_join[~spatial_join.index.duplicated(keep='first')]
            spot_assignment = spatial_join['index_right']
            spot_assignment = spot_assignment.fillna(-1).round().astype(int)
            assignments[i:i+len(spot_assignment)] = spot_assignment
            i += len(spot_assignment)
        
        
    matched = assignments != -1
    point_gdf = point_gdf.iloc[matched].copy()
    point_gdf['cell_id'] = cell_gdf['cell_id'].iloc[assignments[assignments != -1]].values
    
    
    return point_gdf


def _buffer(block, exp_pixel):
    return block.buffer(exp_pixel)


def expand_nuclei(gdf: gpd.GeoDataFrame, pixel_size: float, exp_um=5, plot=False, n_workers=-1) -> gpd.GeoDataFrame:
    """ Expand the nuclei in every direction by `exp_um` um (derived using `pixel_size`)

    Args:
        gdf (gpd.GeoDataFrame): geodataframe of nuclei as polygons
        pixel_size (float): pixel size in um/px for the coordinate system of gdf
        exp_um (int, optional): expansion in um. Defaults to 5.
        plot (bool, optional): whenever to plot the results (will be slow for >1000 nuclei). Defaults to False.
        n_workers (int, optional): number of threads (-1 to use all cpu cores). Defaults to -1.

    Returns:
        gpd.GeoDataFrame: expanded nucleis
    """
    
    from scipy.spatial import Voronoi
    from shapely.geometry import Point, Polygon
    
    exp_pixel = exp_um / pixel_size
    gdf_cell = gdf.copy()
    
    centroids = gdf_cell.centroid
    
    logger.info('Expand nuclei... (can take a few minutes)')
    
    max_workers = get_n_threads(n_workers)
    
    # Use multithreading here because geopandas.buffer releases the GIL
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = len(gdf) // max_workers
        
        chunks = [gdf.iloc[i:i+chunk_size].geometry for i in range(0, len(gdf), chunk_size)]
        
        futures = [executor.submit(_buffer, geom, exp_pixel) for geom in chunks]
        results = [future.result() for future in futures]
    
        
    gdf_cell.geometry = pd.concat(results)
    
    min_x, min_y, max_x, max_y = centroids.total_bounds
    offset = 250
    ghost_points = np.array(
        [
            Point([min_x - offset, min_y - offset]), 
            Point([min_x - offset, max_y + offset]), 
            Point([max_x + offset, max_y + offset]), 
            Point([max_x + offset, min_y - offset])
        ]
    )
    
    points = np.concatenate((centroids.values, ghost_points))
    
    logger.info('Create Voronoi diagram...')
    
    points_series = gpd.GeoSeries(points)
    x = points_series.x
    y = points_series.y
    xy = np.column_stack((x, y))
    vor = Voronoi(xy)
    
    logger.info('Convert Voronoi regions to polygons...')
    
    voronoi_poly = np.array([Polygon([vor.vertices[i] for i in region]) for region in vor.regions])[vor.point_region][:-len(ghost_points)]
    gdf_vor = gpd.GeoDataFrame(geometry=voronoi_poly)
    gdf_vor.index = gdf_cell.index
    
    # Geopandas voronoi_polygons doesnt return polygons in order, use shapely.vornoi_polygons instead
    # TODO ordered will be added in shapely 2.1, uncomment when released
    # Note that the scipy implementation might still offer better results but it will be slower
    # voronoi_poly = gpd.GeoSeries(voronoi_polygons(MultiPoint(points), ordered=True))
    # gdf_vor = gpd.GeoDataFrame(geometry=voronoi_poly).explode().iloc[:-len(ghost_points)]
    # gdf_vor.index = gdf_cell.index
    
    logger.info('Filter invalid polygons...')
    
    invalid_mask = ~gdf_vor.is_valid
    gdf.loc[~invalid_mask, 'geometry'] = gdf.loc[~invalid_mask, 'geometry'].buffer(0)
    invalid_nb = invalid_mask.sum()
    if invalid_nb > 0:
       logger.warning(f'Found {invalid_nb} invalid shapes during nuclei expansion')
    
    logger.info('Intersect Voronoi regions with buffered nuclei...')
    
    inter = gdf_vor.intersection(gdf_cell)
    
    gdf_cell.geometry = inter
    
    gdf_cell.geometry = gdf_cell.union(gdf.loc[~invalid_mask])
    
    if plot:
        import matplotlib.pyplot as plt
        logger.info('Plotting...')
        _, ax = plt.subplots(figsize=(50, 50))

        #gdf_vor.geometry.plot(ax=ax, color='green', alpha=0.5, edgecolor='black', label='Polygons2')
        gdf_cell.plot(ax=ax, color='red', alpha=0.5, edgecolor='black', label='Polygons1')
        gdf.plot(ax=ax, color='grey', alpha=0.5, edgecolor='black', label='Polygons1')

        plt.legend()
        plt.gca().set_aspect('equal')
        plt.savefig('poly2.jpg')
        
    return gdf_cell


def bin_per_cell(
    nuc_seg: Union[str, gpd.GeoDataFrame], 
    bc_matrix: Union[str, sc.AnnData], 
    path_bins_pos: str, 
    pixel_size: float, 
    save_dir: str = None, 
    exp_um = 5, 
    exp_nuclei: bool = True
) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
    """ Bin Visium-hd sub-bins per cell.

    **Deprecated** use hest.readers.pool_bins_visiumhd_per_cell instead

    Args:
        nuc_seg (Union[str, gpd.GeoDataFrame]): nuclei segmentation
        bc_matrix (Union[str, sc.AnnData]): bc_matrix representing Visium-hd bins.
        path_bins_pos (str): path to `tissue_positions.parquet`
        pixel_size (float): pixel size of path_bins_pos in um/px
        save_dir (str, optional): whenever to save to aligned_cells.h5ad. Defaults to None.
        exp_um (int, optional): nuclei expansion in um if exp_nuclei is True. Defaults to 5.
        exp_nuclei (bool, optional): whenever to expand nuclei to derive cells. Defaults to True.

    Returns:
        Tuple[sc.AnnData, gpd.GeoDataFrame]: binned adata and (expended) nuclei
    """
    warnings.warn(
        "bin_per_cell is deprecated and will be removed in a future version. "
        "Please use 'pool_bins_visiumhd_per_cell' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    verify_paths([bc_matrix, path_bins_pos])
    

    nuclei_gdf = read_seg(nuc_seg)
    
    if exp_nuclei:
        cell_gdf = expand_nuclei(nuclei_gdf, pixel_size, exp_um=exp_um)
    else:
        cell_gdf = nuclei_gdf
    
    logger.info('Read bin positions...')
    points_gdf = read_spots_gdf(path_bins_pos)
    
    assignment = assign_spot_to_cell(cell_gdf, points_gdf)
    
    adata = read_adata(bc_matrix)
    
    cell_adata = sum_per_cell(adata, assignment)
    
    if save_dir is not None:
        cell_adata.write_h5ad(os.path.join(save_dir, 'aligned_cells.h5ad'))
    
    return cell_adata, cell_gdf


def sum_per_cell(adata: sc.AnnData, assignment: gpd.GeoDataFrame):
    import scanpy as sc

    logger.info('filter cells...')
    obs = pd.DataFrame(adata.obs_names, columns=['obs_name'])
    obs['obs_index'] = obs.index
    assignment = assignment.merge(obs, how='inner', left_on='barcode', right_on='obs_name')
    obs_index = assignment['obs_index'].values
    adata = adata[obs_index]
    adata.obs['cell_id'] = assignment['cell_id'].values
    
    logger.info('Sum spots per cell...')
    groupby_object = adata.obs.groupby(['cell_id'], observed=True)
    counts = adata.X
    
    # Obtain the number of unique nuclei and the number of genes in the expression data
    N_groups = groupby_object.ngroups
    N_genes = counts.shape[1]

    from scipy.sparse import lil_matrix

    # Initialize a sparse matrix to store the summed gene counts for each nucleus
    summed_counts = lil_matrix((N_groups, N_genes))

    cell_ids = []
    row = 0

    # Iterate over each unique polygon to calculate the sum of gene counts.
    # TODO parallelize
    for cell_id, idx_ in tqdm(groupby_object.indices.items()):
        summed_counts[row] = counts[idx_].sum(0)
        row += 1
        cell_ids.append(cell_id)
        
    cell_adata = sc.AnnData(X=summed_counts.tocsr() ,obs=pd.DataFrame(cell_ids, columns=['cell_id'], index=cell_ids),var=adata.var)
    return cell_adata
