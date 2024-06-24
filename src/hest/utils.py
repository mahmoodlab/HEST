import concurrent.futures
import gzip
import json
import os
import shutil
from enum import Enum
from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

from hest.wsi import WSIPatcher, wsi_factory

try:
    import pyvips
except Exception:
    print("Couldn't import pyvips, verify that libvips is installed on your system")
import scanpy as sc
import tifffile
from kwimage.im_cv2 import imresize
from packaging import version
from PIL import Image
from scipy import sparse
from sklearn.neighbors import KDTree
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 93312000000
ALIGNED_HE_FILENAME = 'aligned_fullres_HE.tif'


def geojson_to_map(geojson: dict, width, height, color=None):
    img = np.zeros((height, width), dtype=np.uint8)
    
    i = 0
    for chunk in geojson:  
        for my_coords in tqdm(chunk['geometry']['coordinates']):
            contours = my_coords[0]
            contours = (np.array(contours)).round().astype(np.int32)
            if color is None:
                c = (i + 1, i + 1, i + 1)
            else:
                c = color
            cv2.drawContours(img, [contours], -1, color=c, thickness=cv2.FILLED)
            i = (i + 1) % 255
    
    #img = cv2.resize(img, (width, height))
    #Image.fromarray(img).save('contours.jpg')
    #tiff_save(img, 'contours.tif', 0.5, pyramidal=True, bigtiff=False)
    return img
        

def to_instance_map(img, cells_mask, save_dir, target_pixel_size, src_pixel_size, target_patch_size) -> None:
    scale_factor = target_pixel_size / src_pixel_size
    patch_size_pxl = round(target_patch_size * scale_factor)
    
    patch_dir = os.path.join(save_dir, 'patches')
    mask_dir = os.path.join(save_dir, 'masks')
    overlay_dir = os.path.join(save_dir, 'overlays')
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    wsi = wsi_factory(img)
    patcher = WSIPatcher(wsi, patch_size_pxl)
    
    cells_mask = wsi_factory(cells_mask)
    
    cols, rows = patcher.get_cols_rows()
    
    print('save patches...')
    # saves patches to temp dir
    for row in tqdm(range(rows)):
        for col in range(cols):
            tile, pxl_x, pxl_y = patcher.get_tile(col, row)
            mask = cells_mask.read_region((pxl_x, pxl_y), (patch_size_pxl, patch_size_pxl))
            mask = mask[:, :, 0]
            mask = cv2.resize(mask, (target_patch_size, target_patch_size), interpolation=cv2.INTER_NEAREST)
            
            tile = cv2.resize(tile, (target_patch_size, target_patch_size), interpolation=cv2.INTER_CUBIC)

            Image.fromarray(tile).save(os.path.join(patch_dir, f'{pxl_x}_{pxl_y}.png'))
            
            
            inst_map = mask.copy()
            mask[mask > 0] = 1
            
            dict = {
                'inst_map': inst_map,
                'type_map': mask
            }
            
            Image.fromarray(inst_map).save(os.path.join(mask_dir, f'{pxl_x}_{pxl_y}.png'))
            np.save(os.path.join(mask_dir, f'{pxl_x}_{pxl_y}.png'), dict)
            blended = np.zeros((target_patch_size, target_patch_size, 3), dtype=np.uint8)
            blended[mask > 0] = [21, 237, 72]
            blended = Image.fromarray(blended)
            blended = Image.blend(Image.fromarray(tile), blended, 0.5)
            blended.save(os.path.join(overlay_dir, f'{pxl_x}_{pxl_y}.png'))
            


def combine_meta_metrics(meta_df, metrics_path, meta_path):
    for _, row in meta_df.iterrows():
        row_dict = row.to_dict()
        with open(os.path.join(metrics_path, row_dict['id'] + '.json')) as f:
            metrics_dict = json.load(f)
        combined_dict = {**metrics_dict, **row_dict}
        with open(os.path.join(meta_path, row_dict['id'] + '.json'), 'w') as f:
            json.dump(combined_dict, f)
            

def df_morph_um_to_pxl(df, x_key, y_key, pixel_size_morph):
    df[x_key] = df[x_key] / pixel_size_morph
    df[y_key] = df[y_key] / pixel_size_morph
    return df


def align_xenium_df(alignment_file_path, pixel_size_morph, df_transcripts, x_key, y_key):
    alignment_file = pd.read_csv(alignment_file_path, header=None)
    alignment_matrix = alignment_file.values
    #convert alignment matrix from pixel to um
    alignment_matrix[0][2] *= pixel_size_morph
    alignment_matrix[1][2] *= pixel_size_morph
    he_to_morph_matrix = alignment_matrix
    alignment_matrix = np.linalg.inv(alignment_matrix)
    coords = np.column_stack((df_transcripts[x_key].values, df_transcripts[y_key].values, np.ones((len(df_transcripts),))))
    aligned = (alignment_matrix @ coords.T).T
    df_transcripts[y_key] = aligned[:,1]
    df_transcripts[x_key] = aligned[:,0]
    return df_transcripts, he_to_morph_matrix, alignment_matrix


def chunk_sorted_df(df, nb_chunk):
    l = len(df) // nb_chunk
    arr = df['cell_id'].values
    i = 0
    coords = []
    while i < len(df):
        start = i
        j = i + l
        while j < len(df):
            if arr[j] != arr[j - 1]:
                break
            j += 1
        end = min(len(df), j)
        coords.append((start, end))
        i = j
    
    chunks = []
    for start, end in coords:
        chunks.append(df[start:end])
    return chunks
    


def read_10x_seg(seg_file: Union[str, pd.DataFrame], type: str = 'Nucleus') -> list:
    
    color = {
        'Cell': [
            255,
            0,
            0
        ],
        'Nucleus': [
            255,
            159,
            68
        ]
    }
    
    print(f"Converting 10x segmentation to geojson ({type})... (can take some time)")
    if isinstance(seg_file, str):
        df = pd.read_parquet(seg_file)
    else:
        df = seg_file

    df['vertex_x'] = df['vertex_x'].astype(float).round(decimals=2)
    df['vertex_y'] = df['vertex_y'].astype(float).round(decimals=2)
    
    df['combined'] = df[['vertex_x', 'vertex_y']].values.tolist()
    df = df[['cell_id', 'combined']]
    
    df['cell_id'], _ = pd.factorize(df['cell_id'])
    aggr_df = df.groupby('cell_id').agg({
        'combined': list
        }
    )
    
    aggr_df['combined'] = [[x] for x in aggr_df['combined']]
    
    coords = list(aggr_df['combined'].values)
    
    ## Shard the cells in n groups to speed up reading in QuPath
    n = 10
    l = round(np.ceil(len(coords) // n))

    cells = []
    for i in range(n):
        start, end = i * l, (i + 1) * l
        end = min(end, len(coords))

        cell = {
            'type': 'Feature',
            'id': type + '-id-' + str(i),
            'geometry': {
                'type': 'MultiPolygon',
                'coordinates': coords[start:end]
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": type + ' ' + str(i),
                    "color": color[type]
                }
            }
        }
        cells.append(cell)
    
    return cells
    

def get_path_relative(file, path) -> str:
    curr_dir = os.path.dirname(os.path.abspath(file))
    return os.path.join(curr_dir, path)


def enc_results_to_table(path) -> str:
    with open(path) as f:
        dict = json.load(f)['results']
    
    first_dataset = dict[0]['results']
    encoders = []
    for encoder in first_dataset:
        encoders.append(encoder['encoder_name'])

    datasets = []
    for d in dict:
        datasets.append(d["dataset_name"])

    df_str = pd.DataFrame("", columns=encoders, index=datasets)
    df_int = pd.DataFrame(0, columns=encoders, index=datasets)
        
    for dataset in dict:
        row = dataset['dataset_name']
        for encoder in dataset['results']:
            col = encoder['encoder_name']
            mean = encoder['pearson_mean']
            std = encoder['pearson_std']
            df_str.loc[row, col] = str(round(mean, 3)).ljust(5, '0') + ' +- ' + str(round(std, 2)).ljust(4, '0')
            df_int.loc[row, col] = mean
    
    df_int.loc['Average'] = round(df_int.mean(axis=0), 3)
    df_str.loc['Average'] = df_int.loc['Average'].astype(str).apply(lambda x: x.ljust(5, '0'))
    
    names = {
        'resnet50_trunc': 'ResNet50',
        'kimianet': 'KimiaNet',
        'ciga': 'Ciga',
        'ctranspath': 'CTransPath',
        'remedis': 'Remedis',
        'phikon_official_hf': 'Phikon',
        'plip': 'PLIP',
        'uni_v1_official': 'UNI',
        'conch_v1_official': 'CONCH',
        'gigapath': 'gigapath'
    }
    
    df_str = df_str[names.keys()]
    df_int = df_int[names.keys()]
    df_str = df_str.rename(columns=names)
    df_int = df_int.rename(columns=names)
    
    df_str.to_csv('str.csv')
    df_int.to_csv('int.csv')
    # with open('latex.txt') as f:
    latex = df_str.to_latex()
    print(latex)
    return latex
    
    
def compare_meta_df(meta_df1, meta_df2):
    diff1 = set(meta_df1['id'].values) - set(meta_df2['id'].values)
    diff2 = set(meta_df2['id'].values) - set(meta_df1['id'].values)
    print('only in meta1: ', diff1)
    print('')
    print('only in meta2: ', diff2)
    return list(diff1), list(diff2)


def normalize_adata(adata: sc.AnnData, scale=1e6, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    
    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            neighbors_index = adata.obs[((adata.obs['array_row'] >= row - 1) & (adata.obs['array_row'] <= row + 1)) & \
                ((adata.obs['array_col'] >= col - 1) & (adata.obs['array_col'] <= col + 1))].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg
    
    
    sc.pp.normalize_total(filtered_adata, target_sum=1, inplace=True)
    # Facilitate training when using the MSE loss. 
    # This'trick' is also used by Y. Zeng et al in "Spatial transcriptomics prediction from histology jointly through Transformer and graph neural networks"
    filtered_adata.X = filtered_adata.X * scale 
    
    # Logarithm of the expression
    sc.pp.log1p(filtered_adata) 

    return filtered_adata


def cp_right_folder(path_df: str) -> None:
    """Given the path to a dataframe containing two columns ['sample_id, 'subseries']
    copy all the files beginning with 'sample_id' to the directory named after 'subseries'

    Args:
        path_df (str): path to the .csv
    """
    df = pd.read_csv(path_df)
    for _, row in df.iterrows():
        root = os.path.dirname(path_df)
        for file in os.listdir(root):
            if not file.startswith(row['sample_id']):
                continue
                
            src = os.path.join(root, file)
            os.makedirs(os.path.join(root, row['subseries']), exist_ok=True)
            dst = os.path.join(root, row['subseries'], file)
            shutil.move(src, dst)

def get_k_genes_from_df(meta_df: pd.DataFrame, k: int, criteria: str, save_dir: str=None) -> List[str]:
    """Get the k genes according to some criteria across common genes in all the samples in the HEST meta dataframe

    Args:
        meta_df (pd.DataFrame): HEST meta dataframe
        k (int): number of genes to return
        criteria (str): criteria for the k genes to return
            - 'mean': return the k genes with the largest mean expression across samples
            - 'var': return the k genes with the largest expression variance across samples
        save_dir (str, optional): genes are saved as json array to this path if not None. Defaults to None.

    Returns:
        List[str]: k genes according to the criteria
    """
    adata_list = []
    for _, row in meta_df.iterrows():
        id = row['id']
        adata = sc.read_h5ad(f"/mnt/sdb1/paul/images/adata/{id}.h5ad")
        adata_list.append(adata)
    return get_k_genes(adata_list, k, criteria, save_dir=save_dir)


def get_k_genes(adata_list: List[sc.AnnData], k: int, criteria: str, save_dir: str=None, min_cells_pct=0.10) -> List[str]:
    """Get the k genes according to some criteria across common genes in all the samples in the adata list

    Args:
        adata_list (List[sc.AnnData]): list of scanpy AnnData containing gene expressions in adata.to_df()
        k (int): number of most genes to return
        criteria (str): criteria for the k genes to return
            - 'mean': return the k genes with the largest mean expression across samples
            - 'var': return the k genes with the largest expression variance across samples
        save_dir (str, optional): genes are saved as json array to this path if not None. Defaults to None.
        min_cells_pct (float): filter out genes that are expressed in less than min_cells_pct% of the spots for each slide

    Returns:
        List[str]: k genes according to the criteria
    """
    
    check_arg(criteria, 'criteria', ['mean', 'var'])
    
    common_genes = None
    stacked_expressions = None

    # Get the common genes
    for adata in adata_list:
        my_adata = adata.copy()
        
        if min_cells_pct:
            print('min_cells is ', np.ceil(min_cells_pct * len(my_adata.obs)))
            sc.pp.filter_genes(my_adata, min_cells=np.ceil(min_cells_pct * len(my_adata.obs)))
        curr_genes = np.array(my_adata.to_df().columns)
        if common_genes is None:
            common_genes = curr_genes
        else:
            common_genes = np.intersect1d(common_genes, curr_genes)
            

    common_genes = [gene for gene in common_genes if 'BLANK' not in gene and 'Control' not in gene]

    for adata in adata_list:

        if stacked_expressions is None:
            stacked_expressions = adata.to_df()[common_genes]
        else:
            stacked_expressions = pd.concat([stacked_expressions, adata.to_df()[common_genes]])

    if criteria == 'mean':
        nb_spots = len(stacked_expressions)
        mean_expression = stacked_expressions.sum() / nb_spots
        
        top_k = mean_expression.nlargest(k).index
    elif criteria == 'var':
        stacked_adata = sc.AnnData(stacked_expressions.astype(np.float32))
        sc.pp.filter_genes(stacked_adata, min_cells=0)
        sc.pp.log1p(stacked_adata)
        sc.pp.highly_variable_genes(stacked_adata, n_top_genes=k)
        top_k = stacked_adata.var_names[stacked_adata.var['highly_variable']][:k].tolist()
    else:
        raise NotImplementedError()

    if save_dir is not None:
        json_dict = {'genes': list(top_k)}
        with open(save_dir, mode='w') as json_file:
            json.dump(json_dict, json_file)

    print(f'selected genes {top_k}')
    return top_k



class SpotPacking(Enum):
    """Types of ST spots disposition, 
    for Orange Crate Packing see:
    https://kb.10xgenomics.com/hc/en-us/articles/360041426992-Where-can-I-find-the-Space-Ranger-barcode-whitelist-and-their-coordinates-on-the-slide    
    """
    ORANGE_CRATE_PACKING = 0
    GRID_PACKING = 1
   

def get_path_from_meta_row(row):
    """ Get raw data path from meta_df row"""
    
    subseries = row['subseries']
    if isinstance(subseries, float):
        subseries = ""
        
    tech = row['st_technology']
    if isinstance(tech, float):
        tech = 'visium'
    elif 'visium' in tech.lower() and ('visium hd' not in tech.lower()):
        tech = 'visium'
    elif 'xenium' in tech.lower():
        tech = 'xenium'
    elif 'spatial transcriptomics' in tech.lower():
        tech = 'ST'
    elif 'visium hd' in tech.lower():
        tech = 'visium-hd'
    else:
        raise Exception(f'unknown tech {tech}')
    path = os.path.join('/mnt/sdb1/paul/data/samples/', tech, row['dataset_title'], subseries)
    return path   

    
def create_joined_gene_plots(meta, gene_plot=False):
    # determine common genes
    if gene_plot:
        plot_dir = 'gene_plots'
    else:
        plot_dir = 'gene_bar_plots'
    common_genes = None
    n = len(meta)
    max_col = 2
    for _, row in meta.iterrows():
        path = get_path_from_meta_row(row)
        gene_files = np.array(os.listdir(os.path.join(path, 'processed', plot_dir)))
        if common_genes is None:
            common_genes = gene_files
        else:
            common_genes = np.intersect1d(common_genes, gene_files)
            
    my_dir = '/mnt/sdb1/paul/gene_plot_IDC_xenium'
    os.makedirs(my_dir, exist_ok=True)
    for gene in tqdm(common_genes):
        if gene_plot:
            fig, axes = plt.subplots(int(np.ceil(n / max_col)), max_col)
        else:
            fig, axes = plt.subplots(n, 1)
        i = 0
        for _, row in meta.iterrows():
            path = get_path_from_meta_row(row)
            gene_path = os.path.join(path, 'processed', plot_dir, gene)
            image = Image.open(gene_path)
            if gene_plot:
                row = i // max_col
                col = i % max_col
                axes[row][col].imshow(image)
                axes[row][col].axis('off')
            else:
                axes[i].imshow(image)
                axes[i].axis('off')
            i += 1
        plt.savefig(os.path.join(my_dir, f'{gene}_subplot.png'), bbox_inches='tight', pad_inches=0, dpi=600)
        plt.subplots_adjust(wspace=0.1)
        plt.close()


def split_join_adata_by_col(path, adata_path, col):
    adata = sc.read_h5ad(os.path.join(path, adata_path))
    samples = np.unique(adata.obs[col])
    for sample in samples:
        sample_adata = adata[adata.obs['orig.ident'] == sample]
        try:
            #write_10X_h5(sample_adata, os.path.join(path, f'{sample}.h5'))
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
            #write_10X_h5(sample_adata, os.path.join(path, f'{sample}.h5'))
        except:
            sample_adata.__dict__['_raw'].__dict__['_var'] = sample_adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
            sample_adata.write_h5ad(os.path.join(path, f'{sample}.h5ad'))
            
            
            
def pool_xenium_by_cell(dir_path: str, cellvit_geojson: pd.DataFrame, pixel_size) -> sc.AnnData:
    
    with open(cellvit_geojson) as f:
        arr = json.load(f)    
        neoplastics = [item for item in arr if item.get('properties', {}).get('classification')['name'] == 'Neoplastic'][0]['geometry']
        neoplastic_coords = neoplastics['coordinates']
        
            
    df_cells = pd.read_parquet(os.path.join(dir_path, 'cells.parquet'))
    
    with open(os.path.join(dir_path, 'gene_panel.json')) as f:
        dic = json.load(f)
        targets = dic['payload']['targets']
        genes = [target['type']['data']['name'] for target in targets]
        
    Y = np.column_stack((df_cells['x_centroid'], df_cells['y_centroid'])) / pixel_size
    y_tree = KDTree(Y)
    neoplastic_matches = y_tree.query(neoplastic_coords, k = 1, return_distance = False)
    
    neoplastic_coords_df = pd.DataFrame(neoplastic_coords, columns=['cellvit_x', 'cellvit_y'])
    neoplastic_coords_df['xenium_idx'] = neoplastic_matches
    df_cells['xenium_idx'] = df_cells.index
    
    unique_cell_index = np.unique(neoplastic_matches)
    
    assert (len(unique_cell_index) / len(neoplastic_matches)) > 0.9
    
    # assign each xenium cell to it s cellvit cell
    cellvit_xenium_joined = pd.merge(neoplastic_coords_df, df_cells, how='inner', on='xenium_idx')
    
    
    df_transcripts = pd.read_parquet(os.path.join(dir_path, 'transcripts.parquet'))
    
    joined = pd.merge(df_transcripts, cellvit_xenium_joined, how='inner', on='cell_id')
    
    genes_bytes = [gene.encode('utf-8') for gene in genes]
    gene_arr = np.zeros((len(unique_cell_index), len(genes)))
    
    cols = pd.Index(genes_bytes).get_indexer(joined['feature_name'])
    rows = pd.Index(unique_cell_index).get_indexer(joined['xenium_idx'])
    
    
    np.add.at(gene_arr, (rows, cols), 1)
    
    gene_df = pd.DataFrame(gene_arr, columns=genes, index=unique_cell_index)
    gene_df['xenium_idx'] = gene_df.index
    gene_df = pd.merge(gene_df, cellvit_xenium_joined, how='inner', on='xenium_idx')
    gene_df = gene_df.rename(columns={'cell_id': 'xenium_cell_id', 'x_centroid': 'xenium_x_centroid', 'y_centroid': 'xenium_y_centroid'})
    
    return gene_df
            
            
def pixel_size_to_mag(pixel_size: float) -> str:
    """ convert pixel size in um/px to a rough magnitude

    Args:
        pixel_size (float): pixel size in um/px

    Returns:
       str: rought magnitude corresponding to the pixel size
    """
    
    if pixel_size <= 0.1:
        return '60x'
    elif 0.1 < pixel_size and pixel_size <= 0.25:
        return '40x'
    elif 0.25 < pixel_size and pixel_size <= 0.5:
        return '40x'
    elif 0.5 < pixel_size and pixel_size <= 1:
        return '20x'
    elif 1 < pixel_size and pixel_size <= 4:
        return '10x'  
    elif 4 < pixel_size :
        return '<10x'
    
    
def _get_nan(cell, col_name):
    val = cell[col_name]
    if isinstance(cell[col_name], float):
        return ''
    else:
        return val
        

def create_meta_release(meta_df: pd.DataFrame, version: version.Version) -> None:
    """create a HEST metadata release

    Args:
        meta_df (pd.DataFrame): meta_df of release
        version (version.Version): version of release

    Raises:
        Exception: if release already exists
    """
    META_RELEASE_DIR = '/mnt/sdb1/paul/meta_releases'
    
    # Exclude private data
    meta_df = meta_df[meta_df['dataset_title'] != 'Bern ST']
    
    metric_subset = [
        'pixel_size_um_embedded', 
        'pixel_size_um_estimated', 
        'fullres_px_width',
        'fullres_px_height',
        'spots_under_tissue',
        'inter_spot_dist',
        'spot_diameter'
    ]
    
    for col in metric_subset:
        meta_df[col] = None
        
    meta_df['inter_spot_dist'] = None
    meta_df['spot_diameter'] = None
    meta_df['nb_genes'] = None
    meta_df['image_filename'] = None
    meta_df['magnification'] = None

    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        path = get_path_from_meta_row(row)
        with open(os.path.join(path, 'processed', 'metrics.json')) as f:
            metrics = json.load(f)
        for col in metric_subset:
            meta_df.loc[index, col] = metrics.get(col)
        #adata = sc.read_h5ad(os.path.join(path, 'processed', 'aligned_adata.h5ad'))
        #nb_genes = len(adata.var_names)
        #metrics['adata_nb_col'] = nb_genes
        #with open(os.path.join(path, 'processed', 'metrics.json'), 'w') as f:
        #    json.dump(metrics, f)
        
        if isinstance(row['treatment_comment'], float) or row['treatment_comment'].strip() == '':
            meta_df.loc[index, 'treatment_comment'] = None
        
        
        meta_df.loc[index, 'image_filename'] = _sample_id_to_filename(row['id'])
        meta_df.loc[index, 'subseries'] = _get_nan(meta_df.loc[index], 'subseries')
        meta_df.loc[index, 'disease_comment'] = _get_nan(meta_df.loc[index], 'disease_comment')
        meta_df.loc[index, 'tissue'] = _get_nan(meta_df.loc[index], 'tissue')
        #meta_df.loc[index, 'subseries'] = _get_nan(meta_df.loc[index], 'tissue') + ' , ' +  _get_nan(meta_df.loc[index], 'disease_comment') + ' , ' + _get_nan(meta_df.loc[index], 'subseries')
        
        meta_df.loc[index, 'magnification'] = pixel_size_to_mag(metrics['pixel_size_um_estimated'])
        
        #TODO remove
        meta_df.loc[index, 'nb_genes'] = metrics['adata_nb_col']
        if meta_df.loc[index, 'st_technology'] == 'Xenium':
            meta_df.loc[index, 'spot_diameter'] = None
            meta_df.loc[index, 'inter_spot_dist'] = None
        elif meta_df.loc[index, 'st_technology'] == 'Visium HD':
            meta_df.loc[index, 'spot_diameter'] = 2
            meta_df.loc[index, 'inter_spot_dist'] = 2
        
        #adata = sc.read_h5ad(os.path.join(path, 'processed', 'aligned_adata.h5ad'))             
        #meta_df.loc[index]['nb_genes'] = len(adata.var_names)
        
        
    version_s = str(version).replace('.', '_')
    release_path = os.path.join(META_RELEASE_DIR, f'HEST_v{version_s}.csv')
    if os.path.exists(release_path):
        raise Exception(f'meta already exists at path {release_path}')
    
    release_col_selection = [
        'dataset_title',
        'id',
        'image_filename',
        'organ',
        'disease_state',
        'oncotree_code',
        'species',
        'patient',
        'st_technology',
        'data_publication_date',
        'license',
        'study_link',
        'download_page_link1',
        'inter_spot_dist',
        'spot_diameter',
        'spots_under_tissue',
        'preservation_method',
        'nb_genes',
        'treatment_comment',
        'pixel_size_um_embedded', 
        'pixel_size_um_estimated', 
        'magnification',
        'fullres_px_width',
        'fullres_px_height',
        'tissue',
        'disease_comment',
        'subseries'
    ]
    #release_col_selection += metric_subset
    meta_df = meta_df[release_col_selection]
    meta_df = meta_df[meta_df['pixel_size_um_estimated'].isna() | meta_df['pixel_size_um_estimated'] < 1.15]
    meta_df = meta_df[(meta_df['species'] == 'Mus musculus') | (meta_df['species'] == 'Homo sapiens')]  
    meta_df.to_csv(release_path, index=False)


def find_first_file_endswith(dir: str, suffix: str, exclude='', anywhere=False) -> str:
    """find first file in directory that ends with suffix

    Args:
        dir (str): path to directory to search in
        suffix (str): suffix to test for
        exclude (str, optional): any files that need to be excluded from search. Defaults to ''.
        anywhere (bool, optional): if 'suffix' can occur anywhere in the filename (not only at the end). Defaults to False.

    Returns:
        str: filename of first match
    """
    files_dir = os.listdir(dir)
    if anywhere:
        matching = [file for file in files_dir if suffix in file and file != exclude]
    else:
        matching = [file for file in files_dir if file.endswith(suffix) and file != exclude]
    if len(matching) == 0:
        return None
    else:
        return os.path.join(dir, matching[0])


def tiff_save(img: np.ndarray, save_path: str, pixel_size: float, pyramidal=True, bigtiff=False) -> None:
    """Save an image stored in a numpy array to the generic tiff format

    Args:
        img (np.ndarray): image stored in a number array, shape must be H x W x C
        save_path (str): full path to tiff (including filename)
        pixel_size (float): pixel size (in um/px) that will be embedded in the tiff
        pyramidal (bool, optional): whenever to save to a pyramidal format (WARNING saving to a pyramidal format is much slower). Defaults to True.
        bigtiff (bool, optional): whenever to save as a generic BigTiff, must be set to true if the resulting image is more than 4.1 GB . Defaults to False.
    """
    
    
    if pyramidal:
        print('saving to pyramidal tiff... can be slow')
        pyvips_img = pyvips.Image.new_from_array(img)

        # save in the generic tiff format readable by both openslide and QuPath
        pyvips_img.tiffsave(
            save_path, 
            bigtiff=bigtiff, 
            pyramid=True, 
            tile=True, 
            tile_width=256, 
            tile_height=256, 
            compression='deflate', 
            resunit=pyvips.enums.ForeignTiffResunit.CM,
            xres=1. / (pixel_size * 1e-4),
            yres=1. / (pixel_size * 1e-4))
    else:
        with tifffile.TiffWriter(save_path, bigtiff=bigtiff) as tif:
            options = dict(
                tile=(256, 256), 
                compression='deflate', 
                resolution=(
                    1. / (pixel_size * 1e-4),
                    1. / (pixel_size * 1e-4),
                    'CENTIMETER'
                ),
            )
            tif.write(img, **options)


def find_biggest_img(path: str) -> str:
    """ find filename of biggest image in the `path` directory
    
    it only looks for images of type ['.tif', '.jpg', '.btf', '.png', '.tiff', '.TIF']
    and exclude the following by default ['aligned_fullres_HE.ome.tif', 'morphology.ome.tif', 'morphology_focus.ome.tif', 'morphology_mip.ome.tif']

    Args:
        path (str): where we look for the biggest image

    Raises:
        Exception: if no image can be found

    Returns:
        str: filename of biggest image in `path` directory
    """
    ACCEPTED_FORMATS = ['.tif', '.jpg', '.btf', '.png', '.tiff', '.TIF']
    biggest_size = -1
    biggest_img_filename = None
    for file in os.listdir(path):
        ls = [s for s in ACCEPTED_FORMATS if file.endswith(s)]
        if len(ls) > 0:
            if file not in ['aligned_fullres_HE.ome.tif', 'morphology.ome.tif', 'morphology_focus.ome.tif', 'morphology_mip.ome.tif']:
                size = os.path.getsize(os.path.join(path, file))
                if size > biggest_size:
                    biggest_img_filename = file
                    biggest_size = size
    if biggest_img_filename is None:
        raise Exception(f"Couldn't find an image automatically, make sure that the folder {path} contains an image of one of these types: {ACCEPTED_FORMATS}")
    return biggest_img_filename


def metric_file_do_dict(metric_file_path):
    """convert a HEST metrics file to a dictionary"""
    
    metrics = pd.read_csv(metric_file_path)
    dict = metrics.to_dict('records')[0]
    return dict
    
      
      
def find_pixel_size_from_spot_coords(my_df: pd.DataFrame, inter_spot_dist: float=100., packing: SpotPacking = SpotPacking.ORANGE_CRATE_PACKING) -> Tuple[float, int]:
    """Estimate the pixel size of an image in um/px given a dataframe containing the spot coordinates in that image

    Args:
        my_df (pd.DataFrame): dataframe containing the coordinates of each spot in an image, it must contain the following columns:
            ['pxl_row_in_fullres', 'pxl_col_in_fullres', 'array_col', 'array_row']
        inter_spot_dist (float, optional): the distance in um between two spots on the same row. Defaults to 100..
        packing (SpotPacking, optional): disposition of the spots on the slide. Defaults to SpotPacking.ORANGE_CRATE_PACKING.

    Raises:
        Exception: if cannot find two spots on the same row

    Returns:
        Tuple[float, int]: approximation of the pixel size in um/px and over how many spots that pixel size was estimated
    """
    def _cart_dist(start_spot, end_spot):
        """cartesian distance in pixel between two spots"""
        d = np.sqrt((start_spot['pxl_col_in_fullres'] - end_spot['pxl_col_in_fullres']) ** 2 \
            + (start_spot['pxl_row_in_fullres'] - end_spot['pxl_row_in_fullres']) ** 2)
        return d
    
    df = my_df.copy()
    
    max_dist_col = 0
    approx_nb = 0
    best_approx = 0
    df = df.sort_values('array_row')
    for _, row in df.iterrows():
        y = row['array_col']
        x = row['array_row']
        if len(df[df['array_row'] == x]) > 1:
            b = df[df['array_row'] == x]['array_col'].idxmax()
            start_spot = row
            end_spot = df.loc[b]
            dist_px = _cart_dist(start_spot, end_spot)
            
            div = 1 if packing == SpotPacking.GRID_PACKING else 2
            dist_col = abs(df.loc[b, 'array_col'] - y) // div
            
            approx_nb += 1
            
            if dist_col > max_dist_col:
                max_dist_col = dist_col
                best_approx = inter_spot_dist / (dist_px / dist_col)
            if approx_nb > 3:
                break
            
    if approx_nb == 0:
        raise Exception("Couldn't find two spots on the same row")
            
    return best_approx, max_dist_col
      

def register_downscale_img(adata: sc.AnnData, img: np.ndarray, pixel_size: float, spot_size=55., target_size=1000) -> Tuple[np.ndarray, float]:
    """ registers a downscale version of `img` and it's corresponding scalefactors to `adata` in adata.uns['spatial']['ST']

    Args:
        adata (sc.AnnData): anndata to which the downscaled image is registered
        img (np.ndarray): full resolution image to downscale
        pixel_size (float): pixel size in um/px of the full resolution image
        spot_size (_type_, optional): spot diameter in um. Defaults to 55..
        target_size (int, optional): downscaled image biggest edge size. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, float]: downscaled image and it's downscale factor from full resolution
    """
    downscale_factor = target_size / np.max(img.shape)
    downscaled_fullres = imresize(img, downscale_factor)
    
    # register the image
    adata.uns['spatial'] = {}
    adata.uns['spatial']['ST'] = {}
    adata.uns['spatial']['ST']['images'] = {}
    adata.uns['spatial']['ST']['images']['downscaled_fullres'] = downscaled_fullres
    adata.uns['spatial']['ST']['scalefactors'] = {}
    adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres'] = spot_size / pixel_size
    adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef'] = downscale_factor
    
    return downscaled_fullres, downscale_factor
            

def _sample_id_to_filename(id):
    return id + '.tif'            

            
def _process_row(dest, row, cp_downscaled: bool, cp_spatial: bool, cp_pyramidal: bool, cp_pixel_vis: bool, cp_adata: bool, cp_meta: bool):
    """ Internal use method, to transfer images to a `release` folder (`dest`)"""
    try:
        path = get_path_from_meta_row(row)
    except Exception:
        print(f'error with path {path}')
        return
    path = os.path.join(path, 'processed')
    if isinstance(row['id'], float):
        my_id = row['id']
        raise Exception(f'invalid sample id {my_id}')
    
    path_fullres = os.path.join(path, 'aligned_fullres_HE.tif')
    if not os.path.exists(path_fullres):
        print(f"couldn't find {path}")
        return
    if cp_pyramidal:
        print(f"create pyramidal tiff for {row['id']}")
        src_pyramidal = os.path.join(path, 'aligned_fullres_HE.tif')
        dst_pyramidal = os.path.join(dest, 'pyramidal', _sample_id_to_filename(row['id']))
        os.makedirs(os.path.join(dest, 'pyramidal'), exist_ok=True)
        shutil.copy(src_pyramidal, dst_pyramidal)
        #dst = os.path.join(dest, 'pyramidal', _sample_id_to_filename(row['id']))
        #bigtiff_option = '' if isinstance(row['bigtiff'], float) or not row['bigtiff']  else '--bigtiff'
        #vips_pyr_cmd = f'vips tiffsave "{path_fullres}" "{dst}" --pyramid --tile --tile-width=256 --tile-height=256 --compression=deflate {bigtiff_option}'
        #subprocess.call(vips_pyr_cmd, shell=True)
        
    if cp_meta:
        path_metrics = os.path.join(path, 'metrics.json')
        os.makedirs(os.path.join(dest, 'metrics'), exist_ok=True)
        path_dest_metrics = os.path.join(dest, 'metrics', row['id'] + '.json')
        shutil.copy(path_metrics, path_dest_metrics)
    if cp_downscaled:
        path_downscaled = os.path.join(path, 'downscaled_fullres.jpeg')
        os.makedirs(os.path.join(dest, 'downscaled'), exist_ok=True)
        path_dest_downscaled = os.path.join(dest, 'downscaled', row['id'] + '_downscaled_fullres.jpeg')
        shutil.copy(path_downscaled, path_dest_downscaled)
    if cp_spatial:
        path_spatial = os.path.join(path, 'spatial_plots.png')
        os.makedirs(os.path.join(dest, 'spatial_plots'), exist_ok=True)
        path_dest_spatial = os.path.join(dest, 'spatial_plots', row['id'] + '_spatial_plots.png')
        shutil.copy(path_spatial, path_dest_spatial)
    if cp_pixel_vis:
        path_pixel_vis = os.path.join(path, 'pixel_size_vis.png')
        os.makedirs(os.path.join(dest, 'pixel_vis'), exist_ok=True)
        path_dest_pixel_vis = os.path.join(dest, 'pixel_vis', row['id'] + '_pixel_size_vis.png')
        if not os.path.exists(path_pixel_vis):
            print(f"couldn't find {path_pixel_vis}")
        else:
            shutil.copy(path_pixel_vis, path_dest_pixel_vis)
    if cp_adata:
        path_adata = os.path.join(path, 'aligned_adata.h5ad')
        os.makedirs(os.path.join(dest, 'adata'), exist_ok=True)
        path_dest_adata = os.path.join(dest, 'adata', row['id'] + '.h5ad')
        shutil.copy(path_adata, path_dest_adata)        
        
            
def copy_processed_images(dest: str, meta_df: pd.DataFrame, cp_spatial=True, cp_downscaled=True, cp_pyramidal=True, cp_pixel_vis=True, cp_adata=True, cp_meta=True, n_job=6):
    """ Internal use method, to transfer images to a `release` folder (`dest`)"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_job) as executor:
        # Submit tasks to the executor
        future_results = [executor.submit(_process_row, dest, row, cp_downscaled, cp_spatial, cp_pyramidal, cp_pixel_vis, cp_adata, cp_meta) for _, row in meta_df.iterrows()]

        # Retrieve results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_results), total=len(meta_df)):
            result = future.result()
    

# taken from https://github.com/scverse/anndata/issues/595
def write_10X_h5(adata, file):
    """Writes adata to a 10X-formatted h5 file.
    
    Note that this function is not fully tested and may not work for all cases.
    It will not write the following keys to the h5 file compared to 10X:
    '_all_tag_keys', 'pattern', 'read', 'sequence'

    Args:
        adata (AnnData object): AnnData object to be written.
        file (str): File name to be written to. If no extension is given, '.h5' is appended.

    Raises:
        FileExistsError: If file already exists.

    Returns:
        None
    """
    
    if '.h5' not in file: file = f'{file}.h5'
    #if os.path.exists(file):
    #    raise FileExistsError(f"There already is a file `{file}`.")
    def int_max(x):
        return int(max(np.floor(len(str(int(np.max(x)))) / 4), 1) * 4)
    def str_max(x):
        return max([len(i) for i in x])

    w = h5py.File(file, 'w')
    grp = w.create_group("matrix")
    grp.create_dataset("barcodes", data=np.array(adata.obs_names, dtype=f'|S{str_max(adata.obs_names)}'))
    grp.create_dataset("data", data=np.array(adata.X.data, dtype=f'<i{int_max(adata.X.data)}'))
    ftrs = grp.create_group("features")
    # this group will lack the following keys:
    # '_all_tag_keys', 'feature_type', 'genome', 'id', 'name', 'pattern', 'read', 'sequence'
    if 'feature_types' not in adata.var:
        adata.var['feature_types'] = ['Unspecified' for _ in range(len(adata.var))]   
    ftrs.create_dataset("feature_type", data=np.array(adata.var.feature_types, dtype=f'|S{str_max(adata.var.feature_types)}'))
    if 'genome' not in adata.var:
        adata.var['genome'] = ['Unspecified_genone' for _ in range(len(adata.var))]
    ftrs.create_dataset("genome", data=np.array(adata.var.genome, dtype=f'|S{str_max(adata.var.genome)}'))
    if 'gene_ids' not in adata.var:
        adata.var['gene_ids'] = ['Unspecified_gene_id' for _ in range(len(adata.var))]
    ftrs.create_dataset("id", data=np.array(adata.var.gene_ids, dtype=f'|S{str_max(adata.var.gene_ids)}'))
    ftrs.create_dataset("name", data=np.array(adata.var.index, dtype=f'|S{str_max(adata.var.index)}'))
    if not isinstance(adata.X, sparse._csc.csc_matrix):
        adata.X = sparse.csr_matrix(adata.X)
    grp.create_dataset("indices", data=np.array(adata.X.indices, dtype=f'<i{int_max(adata.X.indices)}'))
    grp.create_dataset("indptr", data=np.array(adata.X.indptr, dtype=f'<i{int_max(adata.X.indptr)}'))
    grp.create_dataset("shape", data=np.array(list(adata.X.shape)[::-1], dtype=f'<i{int_max(adata.X.shape)}'))


def check_arg(arg, arg_name, values):
    if arg not in values:
       raise ValueError(f"{arg_name}={arg} must be one of {values}")
        

def helper_mex(path: str, filename: str) -> None:
    """If filename doesn't exist in directory `path`, find similar filename in same directory and zip it to patch filename"""
    # zip if needed
    file = os.path.join(path, filename.strip('.gz'))
    dst = os.path.join(path, filename)
    src = os.path.join(path, filename)
    if file is not None and src is None:
        f_in = open(file, 'rb')
        f_out = gzip.open(os.path.join(os.path.join(path), filename), 'wb')
        f_out.writelines(f_in)
        f_out.close()
        f_in.close()
    
    if not os.path.exists(dst) and \
            src is not None:
        shutil.copy(src, dst)


def load_image(img_path: str) -> Tuple[np.ndarray, float]:
    """Load image from path and it's corresponding embedded pixel size in um/px
    
    the embedded pixel size is only determined in tiff/tif/btf/TIF images and
    only if the tags 'XResolution' and 'YResolution' are set

    Args:
        img_path (str): path to image

    Returns:
        Tuple[np.ndarray, float]: image and it's embedded pixel size in um/px
    """
    unit_to_micrometers = {
        tifffile.RESUNIT.INCH: 25.4,
        tifffile.RESUNIT.CENTIMETER: 1.e4,
        tifffile.RESUNIT.MILLIMETER: 1.e3,
        tifffile.RESUNIT.MICROMETER: 1.,
        tifffile.RESUNIT.NONE: 1.
    }
    pixel_size_embedded = None
    if img_path.endswith('tiff') or img_path.endswith('tif') or img_path.endswith('btf') or img_path.endswith('TIF'):
        img = tifffile.imread(img_path)
            
        
        my_img = tifffile.TiffFile(img_path)
        
        if 'XResolution' in my_img.pages[0].tags and my_img.pages[0].tags['XResolution'].value[0] != 0 and 'ResolutionUnit' in my_img.pages[0].tags:
            # result in micrometers per pixel
            factor = unit_to_micrometers[my_img.pages[0].tags['ResolutionUnit'].value]
            pixel_size_embedded = (my_img.pages[0].tags['XResolution'].value[1] / my_img.pages[0].tags['XResolution'].value[0]) * factor
    else:
        img = np.array(Image.open(img_path))
        
    # sometimes the RGB axis are inverted
    if img.shape[0] == 3:
        img = np.transpose(img, axes=(1, 2, 0))
    elif img.shape[2] == 4: # RGBA to RGB
        img = img[:,:,:3]
    if np.max(img) > 1000:
        img = img.astype(np.float32)
        img /= 2**8
        img = img.astype(np.uint8)
    
    return img, pixel_size_embedded


def _plot_center_square(width, height, length, color, text, offset=0):
    """Plot square centered in plot"""
    if length > width * 4 and length > height * 4:
        return
    margin_x = (width - length) / 2
    margin_y = (height - length) / 2
    plt.text(width // 2, (height // 2) + offset, text, fontsize=12, ha='center', va='center', color=color)
    plt.plot([margin_x, length + margin_x], [margin_y, margin_y], color=color)
    plt.plot([margin_x, length + margin_x], [height - margin_y, height - margin_y], color=color)
    plt.plot([margin_x, margin_x], [margin_y, height - margin_y], color=color)
    plt.plot([length + margin_x, length + margin_x], [margin_y, height - margin_y], color=color)


def plot_verify_pixel_size(downscaled_img: np.ndarray, down_fact: float, pixel_size_embedded: float, pixel_size_estimated: float, path: float) -> None:
    """Plot squares on a downscaled image for scale comparison"""
    plt.imshow(downscaled_img)

    width = downscaled_img.shape[1]
    height = downscaled_img.shape[0]
    
    if pixel_size_embedded is not None:
        length_embedded = (6500. / pixel_size_embedded) * down_fact
        _plot_center_square(width, height, length_embedded, 'red', '6.5m, embedded', offset=50)
        
    if pixel_size_estimated is not None:
        length_estimated = (6500. / pixel_size_estimated) * down_fact
        _plot_center_square(width, height, length_estimated, 'blue', '6.5m, estimated')
    
    plt.savefig(path)
    plt.close()



def save_scalefactors(adata: sc.AnnData, path) -> None:
    """Save scale factors to path from adata.uns"""
    dict = {}
    dict['tissue_downscaled_fullres_scalef'] = adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']
    dict['spot_diameter_fullres'] = adata.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']
    
    with open(path, 'w') as json_file:
        json.dump(dict, json_file)
