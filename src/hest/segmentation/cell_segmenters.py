from __future__ import annotations

import os
import sys
import traceback
import warnings
from abc import abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
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
        
    
    
class CellViTSegmenter():
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
        wsi_paths: "{wsi_path}"
        wsi_properties:
            magnification: 40
            slide_mpp: {src_pixel_size}
        batch_size: {batch_size}
        """
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = 'debug_seg'
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
        
    
    def segment_cells(
        self, 
        wsi_path: str, 
        name: str, 
        src_pixel_size: float=None, 
        dst_pixel_size: float=0.25, 
        batch_size=2, 
        gpu_ids=[], 
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
        
        preprocess_path = self._preprocess(wsi_path, name, src_pixel_size, dst_pixel_size, save_dir)
        
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
        dst_pixel_size, 
        batch_size, 
        gpu_ids, 
        save_dir,
        model
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
    
    
def assign_to_cell(cell_gdf, point_gdf):
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    # shard points dataframe
    n = 1000
    l = round(np.ceil(len(point_gdf) / n))
    logger.info('matching spots to cells... (can take a few minutes)')
    assignments = np.zeros(len(point_gdf), dtype=int)
    for i in tqdm(range(n)):
        start = l * i
        end = min(l * (i + 1), len(point_gdf))
        
        
        gdf_points_shard = point_gdf[start:end]
        
    
        spatial_join = gpd.sjoin(gdf_points_shard, cell_gdf, how='left', predicate='within')
        spatial_join = spatial_join[~spatial_join.index.duplicated(keep='first')]
        spot_assignment = spatial_join['index_right']
        spot_assignment = spot_assignment.fillna(-1).round().astype(int)
        assignments[start:end] = spot_assignment
        
    point_gdf['index_right'] = assignments
    # Match index to cell_id
    point_gdf['cell_id'] = point_gdf.merge(cell_gdf, left_on='index_right', right_index=True, how='left')['cell_id']
    
    point_gdf = point_gdf.dropna(subset=['cell_id'])
    point_gdf['cell_id'] = point_gdf['cell_id'].astype(int)
    
    return point_gdf


def _buffer(geom, exp_pixel):
    centroids = [poly.centroid.coords[0] for poly in geom['geometry']]
    return [geom.buffer(exp_pixel), centroids]


def _rm_invalid(geom, box):
    geom = geom['geometry']
    invalid = [geo.buffer(0) for geo in geom if not geo.is_valid]
    
    geom = geom.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    geom = gpd.GeoDataFrame(geometry=[geo.intersection(box) for geo in geom])
    return [geom, len(invalid)]
    


def expand_nuclei(gdf, pixel_size, exp_um, plot=False):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from shapely.geometry import Point, Polygon
    
    gdf = gdf[:len(gdf)]
    
    exp_pixel = exp_um / pixel_size
    logger.info('Expand nuclei... (can take a few minutes)')
    gdf_cell = gdf.copy()
    
    with Pool(cpu_count()) as pool:
        chunk_size = len(gdf) // cpu_count()
        chunks = [gdf.iloc[i:i+chunk_size] for i in range(0, len(gdf), chunk_size)]
    
        res = pool.starmap(_buffer, [(geom, exp_pixel) for geom in chunks])
        results, centroids = [i[0] for i in res], [i[1] for i in res]
        
    gdf_cell = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
    
    points = np.concatenate(centroids)
    
    ghost_points = [Point(0, 0), Point(0, 20000), Point(20000, 20000), Point(20000, 0)]
    
    points = np.concatenate((points, ghost_points))
    
    logger.info('Create Voronoi diagram...')
    
    vor = Voronoi(points)
    
    #voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
    
    logger.info('Convert Voronoi regions to polygons...')
    
    voronoi_poly = [Polygon([vor.vertices[i] for i in region]) for region in vor.regions]
    voronoi_poly = np.array(voronoi_poly)[vor.point_region]
    
    
    gdf_vor = gpd.GeoDataFrame(geometry=voronoi_poly)
    
    logger.info('Filter invalid polygons...')

    
    fig, ax = plt.subplots(figsize=(50, 50))
    
    #invalid = gpd.GeoDataFrame(geometry=invalid)
    
    min_x = min(points[:, 0])
    min_y = min(points[:, 1])
    max_x = max(points[:, 0])
    max_y = max(points[:, 1])
    
    offset = 200
    
    box = Polygon([
        (min_x - offset, min_y - offset), 
        (min_x - offset, max_y + offset), 
        (max_x + offset, max_y + offset), 
        (max_x + offset, min_y - offset)
    ])
    

    with Pool(cpu_count()) as pool:
        chunk_size = len(gdf_vor) // cpu_count()
        chunks = [gdf_vor.iloc[i:i+chunk_size] for i in range(0, len(gdf_vor), chunk_size)]
    
        res = pool.starmap(_rm_invalid, [(geom, box) for geom in chunks])
        
    gdf_vor = gpd.GeoDataFrame(pd.concat([i[0] for i in res], ignore_index=True))
    invalid_nb = np.array([i[1] for i in res]).sum()
    if invalid_nb > 0:
        logger.warning(f'Found {invalid_nb} invalid shapes')
    
    logger.info('Intersect Voronoi regions with buffered nuclei...')
    
    gdf_cell = gdf_cell.set_geometry(0)
    inter = gdf_vor.intersection(gdf_cell)
    
    inter = inter[:-len(ghost_points)]
    gdf_cell = gdf.copy()
    gdf_cell.geometry = inter.geometry
    
    if plot:
        logger.info('Plotting...')

        inter.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black', label='Polygons1')
        gdf.plot(ax=ax, color='green', alpha=0.5, edgecolor='black', label='Polygons2')

        plt.legend()
        plt.gca().set_aspect('equal')
        plt.savefig('poly.jpg')
        
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
) -> sc.AnnData:
    verify_paths_exist(paths=[nuc_seg, bc_matrix, path_bins_pos])
    
    nuclei_gdf = read_seg(nuc_seg)
    
    if exp_nuclei:
        cell_gdf = expand_nuclei(nuclei_gdf, pixel_size, exp_um=exp_um)
    else:
        cell_gdf = nuclei_gdf
    
    logger.info('Read bin positions...')
    points_gdf = read_spots_gdf(path_bins_pos)
    
    assignment = assign_to_cell(cell_gdf, points_gdf)
    
    adata = read_adata(bc_matrix)
    
    cell_adata = sum_per_cell(adata, assignment)
    
    if save_dir is not None:
        cell_adata.write_h5ad(os.path.join(save_dir, name + 'cell_bin.h5ad'))
    
    return cell_adata


def sum_per_cell(adata: sc.AnnData, assignment: gpd.GeoDataFrame):

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


def warp_gdf_old(gdf: gpd.GeoDataFrame, displacement_field, field_factor, pad_src) -> gpd.GeoDataFrame:
    pad_src_top = pad_src[0][0]
    pad_src_bottom = pad_src[0][1]
    pad_src_left = pad_src[1][0]
    pad_src_right = pad_src[1][1]
    dis_height = displacement_field.shape[0]
    dis_width = displacement_field.shape[1]
    
    def swap(yx):
        return yx[1], yx[0]
    
    def warp_polygon(polygon):
        try:
            return Polygon(
                [
                    list(swap(displacement_field[pad_src_top + round(y / 5), pad_src_left + round(x / 5)] * 5 + np.array([y, x])))
                    for x, y in polygon.exterior.coords 
                ]
            )
        except Exception:
            print('skip')
            return float('NaN')
    
    gdf.geometry = [warp_polygon(polygon) for polygon in tqdm(gdf.geometry)]
    return gdf


def warp_gdf(gdf: gpd.GeoDataFrame, slide_obj, slide_obj_target) -> gpd.GeoDataFrame:
    
    def warp_polygon(polygon):
        try:
            return Polygon(
                [
                    slide_obj.warp_xy([[x, y]])
                    for x, y in polygon.exterior.coords 
                ]
            )
        except Exception:
            print('skip')
            return float('NaN')
        
    gdf['points'] = [list(polygon.exterior.coords) for polygon in gdf.geometry]
    point_gdf = gdf.explode('points')
    point_gdf['points'] = point_gdf['points'].apply(lambda x: list(x))
    warped = slide_obj.warp_xy_from_to(list(point_gdf['points'].values), to_slide_obj=slide_obj_target)
    point_gdf['warped'] = warped.tolist()

    aggr_df = point_gdf.groupby('cell_id').agg({
        'warped': list
        }
    )
    
    polygons = [Polygon(x) for x in aggr_df['warped']]
    
    
    gdf.geometry = polygons
    return gdf


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