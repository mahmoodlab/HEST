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
import openslide
import pandas as pd
from hestcore.segmentation import get_path_relative
from hestcore.wsi import wsi_factory
from loguru import logger
from shapely import Polygon
from shapely.affinity import translate
from tqdm import tqdm

from hest.io.seg_readers import GeojsonCellReader
from hest.utils import deprecated, get_n_threads, verify_paths
from hestcore.wsi import wsi_factory
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
    models = {
        'CellViT-SAM-H-x40.pth': 'https://drive.google.com/uc?id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q',
        'CellViT-SAM-H-x20.pth': 'https://drive.google.com/uc?id=1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30'
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
        wsi_paths: {wsi_path}
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
    
    
    def _verify_model(self, model_path):
        import gdown
        
        if not os.path.exists(model_path):
            print(f'Model not found at {model_path}, downloading...')
            url = 'https://drive.google.com/uc?id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q'
            gdown.download(url, model_path, quiet=False)
        else:
            print(f'Found model at {model_path}')
        
    
    def _segment_cells_imp(
        self, 
        wsi_path: str, 
        name: str, 
        pixel_size: float, 
        dst_pixel_size: float=0.25, 
        batch_size=2, 
        gpu_ids=[0], 
        save_dir='results/segmentation',
        model='CellViT-SAM-H-x40.pth'
    ) -> str:
        try:
            import cellvit_light
        except:
            print(traceback.format_exc())
            cellvit_light_error()
            
        verify_paths([wsi_path])
        
        model_dir = get_path_relative(__file__, '../../../models')
        model_path = os.path.join(model_dir, model)
        
        if model == 'CellViT-SAM-H-x40.pth':
            self._verify_model(model_path)
        else:
            if not os.path.exits(model_path):
                raise Exception("Can't find model weights {model_path}, only 'CellViT-SAM-H-x40.pth' can be downloaded automatically")
        
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
    dst_pixel_size: float=0.25, 
    batch_size=2, 
    gpu_ids=[0], 
    save_dir='results/segmentation',
    model='CellViT-SAM-H-x40.pth'
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
        model (str, optional): name of model weights to use. Defaults to 'CellViT-SAM-H-x40.pth'.
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
    name = '',
    exp_um = 5, 
    exp_nuclei: bool = True
) -> Tuple[sc.AnnData, gpd.GeoDataFrame]:
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


class AlignmentRefiner:
    
    @abstractmethod
    def refine(self, centroids: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, class_key='class') -> gpd.GeoDataFrame:
        pass

class RegAlignmentRefiner:
    
    def refine(self, centroids: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, class_key='class') -> gpd.GeoDataFrame:
        pass
    
    
class SegAlignmentRefiner(AlignmentRefiner):
    
    def refine(self, centroids: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, class_key='class') -> gpd.GeoDataFrame:
        merged = centroids.sjoin(polygons, how='inner', predicate='within')
        point_in_poly_idx = merged.index
        poly_cont_point_idx = merged['index_right']
        
        
        filt_centroids = centroids.drop(point_in_poly_idx)
        filt_poly = polygons.drop(poly_cont_point_idx)
        
        nearest_idx = filt_centroids.geometry.centroid.sindex.nearest(filt_poly.geometry)[1]
        filt_poly[class_key] = filt_centroids.iloc[nearest_idx][class_key].values
        
        matched_poly = polygons.loc[poly_cont_point_idx]
        cont = matched_poly.sjoin(centroids, how='left', predicate='contains')
        cont = cont.drop_duplicates(keep='last')
        matched_poly[class_key] = cont[f'{class_key}_right'].values
        
        gdf = gpd.GeoDataFrame(pd.concat([matched_poly, filt_poly], ignore_index=True))
        
        return gdf
    
    
def alignment_refiner_factory(method) -> AlignmentRefiner:
    if method == 'seg':
        return SegAlignmentRefiner()
    elif method == 'reg':
        return RegAlignmentRefiner()
    
    
def refine_alignment_reg(
    source_path: str, 
    target_path: str,
    output_dir: str,
):  
    
    
    import deeperhistreg

    ### Define Params ###
    registration_params : dict = deeperhistreg.configs.default_initial_nonrigid()
    # Alternative: # registration_params = deeperhistreg.configs.load_parameters(config_path) # To load config from JSON file
    save_displacement_field : bool = True # Whether to save the displacement field (e.g. for further landmarks/segmentation warping)
    copy_target : bool = True # Whether to copy the target (e.g. to simplify the further analysis)
    delete_temporary_results : bool = True # Whether to keep the temporary results
    case_name : str = "Example_Nonrigid" # Used only if the temporary_path is important, otherwise - provide whatever
    temporary_path = None # Will use default if set to None

    ### Create Config ###
    config = dict()
    config['source_path'] = source_path
    config['target_path'] = target_path
    config['output_path'] = output_dir
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['save_displacement_field'] = save_displacement_field
    config['copy_target'] = copy_target
    config['delete_temporary_results'] = delete_temporary_results
    config['temporary_path'] = temporary_path
    
    ### Run Registration ###
    deeperhistreg.run_registration(**config)
  
    
    
def refine_alignment(centroids, polygons, class_key='class', method='seg') -> gpd.GeoDataFrame:
    return alignment_refiner_factory(method).refine(centroids, polygons, class_key=class_key)

@deprecated
def refine_with_anchor(
    gdf: gpd.GeoDataFrame, 
    anchor_gdf: gpd.GeoDataFrame, 
    pixel_size: float, 
    img: Union[str, np.ndarray, openslide.OpenSlide, CuImage],  # type: ignore
    patch_size_um=200,
    max_offset=5, 
    lower_cut=0.4, 
    upper_cut=0.6
) -> gpd.GeoDataFrame:
    
    if isinstance(anchor_gdf.geometry[0], Polygon):
        logger.warning('anchor_gdf should contain points, found polygons, converting to their centroid')
        anchor_gdf = anchor_gdf.copy().centroid
    
    wsi = wsi_factory(img)
    width, height = wsi.get_dimensions()
    patch_size_pxl = patch_size_um / pixel_size
    n_col = round(np.ceil(width / patch_size_pxl))
    n_row = round(np.ceil(height / patch_size_pxl))
    
    center_gdf = gdf.copy()
    center_gdf.geometry = center_gdf.geometry.centroid
    center_gdf['polygons'] = gdf.geometry
    
    ## TODO should match anchors to xenium cells not the other way around
    
    # get nearest anchor for every cell
    nearest_idx = center_gdf.geometry.centroid.sindex.nearest(anchor_gdf.geometry)[1]
    # index of nearest cell for each anchor
    center_gdf = center_gdf.iloc[nearest_idx].reset_index()
    
    #center_gdf['nearest_idx'] = nearest_idx
    
    points_anchor = anchor_gdf.geometry
    center_gdf['offset_x'] = points_anchor.x.values - center_gdf.geometry.x.values
    center_gdf['offset_y'] = points_anchor.y.values - center_gdf.geometry.y.values
    
    
    polygons = []
    
    for i in range(n_row):
        for j in range(n_col):
            x_left = j * patch_size_pxl
            x_right = x_left + patch_size_pxl
            y_top = i * patch_size_pxl
            y_bottom = y_top + patch_size_pxl
            polygons.append(Polygon([(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)]))
            
    grid = gpd.GeoDataFrame(geometry=polygons)
    
    joined = gpd.sjoin(grid, center_gdf, how='left', predicate='contains')
    joined = joined.dropna()
    joined['index_col'] = joined.index
    joined = joined.rename(columns={
        'index_right': 'index_centroid'
    })
    
    def trimmed_mean(series):
        q1 = series.quantile(lower_cut)
        q3 = series.quantile(upper_cut)
        filtered_series = series[(series > q1) & (series < q3)]
        mean_value = filtered_series.mean()
        return mean_value
    
    logger.info('Remove anchor outliers...')
    
    grouped = joined.groupby('index_col').agg({
        'offset_x': trimmed_mean,
        'offset_y': trimmed_mean
    })
    grouped = grouped.fillna(0)
    
    
    joined['pooled_offset_x'] = grouped['offset_x'].clip(-max_offset, max_offset)
    joined['pooled_offset_y'] = grouped['offset_y'].clip(-max_offset, max_offset)
    
    joined = joined.fillna(0)
    joined['class'] = joined.index
    
    def shift_polygon(row):
        return translate(row.geometry, xoff=row['pooled_offset_x'], yoff=row['pooled_offset_y'])


    #joined.geometry = joined.apply(shift_polygon, axis=1)
    
    joined.geometry = center_gdf.geometry.iloc[joined['index_centroid'].astype(int).values].values
    joined['class'] = (joined['pooled_offset_x'] + joined['pooled_offset_y']).round()
    
    logger.info('Shift polygons...')
    
    joined.geometry = joined['polygons']
    
    offsets = joined[~joined.index.duplicated(keep='first')]
    
    gdf_copy = gdf.copy()
    gdf_copy['orig_polygons'] = gdf_copy.geometry
    gdf_copy.geometry = gdf_copy.geometry.centroid
    ## Add unmatched cells back into the dataframe and set offset based on neighbors
    all_cells = gpd.sjoin(grid, gdf_copy, how='left', predicate='contains').dropna().merge(offsets, left_index=True, right_index=True, how='inner')
    all_cells = gpd.GeoDataFrame(all_cells, geometry=all_cells['orig_polygons'])
    all_cells['class'] = all_cells['class_y']
    
    
    all_cells.geometry = all_cells.apply(shift_polygon, axis=1)
    
    
    all_cells = all_cells.drop(columns=['offset_x', 'offset_y', 'pooled_offset_x', 'pooled_offset_y'])
    
    return all_cells