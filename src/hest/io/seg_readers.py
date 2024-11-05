from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
from abc import abstractmethod

import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
from shapely.geometry.polygon import Point, Polygon
from tqdm import tqdm

from hest.utils import align_xenium_df, get_n_threads


def _process(x, extra_props, index_key, class_name):
    from shapely.geometry.polygon import Point, Polygon
    
    geom_type = x['geometry']['type']
    if geom_type == 'MultiPoint':
        coords = [Point(x['geometry']['coordinates'][i]) for i in range(len(x['geometry']['coordinates']))]
    elif geom_type == 'MultiPolygon':
        coords = [Polygon(x['geometry']['coordinates'][i][0]) for i in range(len(x['geometry']['coordinates']))]
    else:
        raise ValueError("Doesn't recognize type {geom_type}, must be either MultiPoint or MultiPolygon")
    
    name = x['properties']['classification']['name']
    
    gdf = gpd.GeoDataFrame(geometry=coords)
    
    class_index = 'class' if not class_name else class_name
    gdf[class_index] = [name for _ in range(len(gdf))]
    
    if index_key is not None:
        indices = x['properties'][index_key]
        values = np.zeros(len(x['geometry']['coordinates']), dtype=bool)
        values[indices] = True
        gdf[index_key] = values
    
    if extra_props:
        extra_props = [k for k in x['properties'].keys() if k not in ['objectType', 'classification']]
        for prop in extra_props:
            val = x['properties'][prop]
            gdf[prop] = [val for _ in range(len(gdf))]
        
    return gdf


def _read_geojson(path, class_name=None, extra_props=False, index_key=None) -> gpd.GeoDataFrame:
    with open(path) as f:
        ls = json.load(f)
        
        sub_gdfs = []
        for x in tqdm(ls):
            sub_gdfs.append(_process(x, extra_props, index_key, class_name))

        gdf = gpd.GeoDataFrame(pd.concat(sub_gdfs, ignore_index=True))
        
    return gdf


class GDFReader:
    @abstractmethod
    def read_gdf(self, path) -> gpd.GeoDataFrame:
        pass
   
def fn(block, i):
    logger.debug(f'start fn block {i}')
    groups = defaultdict(lambda: [])
    [groups[row[0]].append(row[1]) for row in block]
    g = np.array([Polygon(value) for _, value in groups.items()])
    key = np.array([key for key, _ in groups.items()])
    logger.debug(f'finish fn block {i}')
    return np.column_stack((key, g))
    
def groupby_shape(df, col, n_threads, col_shape='xy'):
    n_chunks = n_threads
    
    if n_threads >= 1:
        l = len(df) // n_chunks
        start = 0
        chunk_lens = []
        while start < len(df):
            end = min(start + l, len(df))
            while end < len(df) and df.iloc[end][col] == df.iloc[end - 1][col]:
                end += 1
            chunk_lens.append((start, end))
            start = end 
        
        dfs = []
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            future_results = [executor.submit(fn, df[[col, col_shape]].iloc[start:end].values, start) for start, end in chunk_lens]

            for future in as_completed(future_results):
                dfs.append(future.result()) 
        
        concat = np.concatenate(dfs)
    else:
        concat = fn(df[[col, col_shape]].values, 0)

    gdf = gpd.GeoDataFrame(geometry=concat[:, 1])
    gdf.index = concat[:, 0]
    
    return gdf

class XeniumParquetCellReader(GDFReader):
    
    def __init__(self, pixel_size_morph=None, alignment_matrix=None):
        self.pixel_size_morph = pixel_size_morph
        self.alignment_matrix = alignment_matrix
    
    def read_gdf(self, path, n_workers=0) -> gpd.GeoDataFrame:
        
        df = pd.read_parquet(path)

        
        
        if self.alignment_matrix is not None:
            df = align_xenium_df(
                df,
                self.alignment_matrix, 
                self.pixel_size_morph,  
                'vertex_x', 
                'vertex_y',
                x_key_dist='vertex_x',
                y_key_dist='vertex_y')
        else:
            df['vertex_x'], df['vertex_y'] = df['vertex_x'] / self.pixel_size_morph, df['vertex_y'] / self.pixel_size_morph 

        df['xy'] = list(zip(df['vertex_x'], df['vertex_y']))
        df = df.drop(['vertex_x', 'vertex_y'], axis=1)   
        
        n_threads = get_n_threads(n_workers)
        
        gdf = groupby_shape(df, 'cell_id', n_threads)
        return gdf

class GDFParquetCellReader(GDFReader):
    
    def read_gdf(self, path) -> gpd.GeoDataFrame:
        return gpd.read_parquet(path)


class GeojsonCellReader(GDFReader):
    
    def read_gdf(self, path) -> gpd.GeoDataFrame:
        gdf = _read_geojson(path)
        gdf['cell_id'] = np.arange(len(gdf))
            
        return gdf
    

class TissueContourReader(GDFReader):

    def read_gdf(self, path) -> gpd.GeoDataFrame:      
        gdf = _read_geojson(path, 'tissue_id', extra_props=False, index_key='hole')
        return gdf
    

def write_geojson(gdf: gpd.GeoDataFrame, path: str, category_key: str, extra_prop=False, uniform_prop=True, index_key: str=None, chunk=False) -> None:
        
    if isinstance(gdf.geometry.iloc[0], Point):
        geometry = 'MultiPoint'
    elif isinstance(gdf.geometry.iloc[0], Polygon):
        geometry = 'MultiPolygon'
    else:
        raise ValueError(f"gdf.geometry[0] must be of type Point or Polygon, got {type(gdf.geometry.iloc[0])}")
    
    
    if chunk:
        n = 10
        l = (len(gdf) // n) + 1
        s = []
        for i in range(n):
            s.append(np.repeat(i, l))
        cls = np.concatenate(s)
        
        gdf['_chunked'] = cls[:len(gdf)]
        category_key = '_chunked'
    
    
    groups = np.unique(gdf[category_key])
    colors = generate_colors(groups)
    cells = []
    for group in tqdm(groups):

        slice = gdf[gdf[category_key] == group]
        shapes = slice.geometry
        
        properties = {
            "objectType": "annotation",
            "classification": {
                "name": str(group),
                "color": colors[group]
            }
        }
        
        if extra_prop:
            props = {}
            col_exclude = [category_key, 'geometry']
            if index_key is not None:
                col_exclude.append(index_key)
            for col in [c for c in gdf.columns if c not in col_exclude]:
                if uniform_prop:
                    unique = np.unique(slice[col])
                    if len(unique) != 1:
                        warnings.warn(f"extra property {col} is not uniform for group {group}, found {unique}")
                props[col] = slice[col].iloc[0]
            
            properties = {**properties, **props}
        
        if index_key is not None:
            key = index_key
            props = {}
            mask = (slice[key] == True).values
            props = {key: np.arange(len(mask))[mask].tolist()}
            properties = {**properties, **props}
        
        if isinstance(gdf.geometry.iloc[0], Point):
            shapes = [[point.x, point.y] for point in shapes]
        elif isinstance(gdf.geometry.iloc[0], Polygon):
            shapes = [[[[x, y] for x, y in polygon.exterior.coords]] for polygon in shapes]
        cell = {
            'type': 'Feature',
            'id': (str(id(path)) + '-id-' + str(group)).replace('.', '-'),
            'geometry': {
                'type': geometry,
                'coordinates': shapes
            },
            "properties": properties
        }
        cells.append(cell)
    
    with open(path, 'w') as f:
        json.dump(cells, f, indent=4)
            
    
    
def generate_colors(names):
    from matplotlib import pyplot as plt
    colors = plt.get_cmap('hsv', len(names))
    color_dict = {}
    for i in range(len(names)):
        rgb = colors(i)[:3]
        rgb = [int(255 * c) for c in rgb]
        color_dict[names[i]] = rgb
    return color_dict


def read_parquet_schema_df(path: str) -> pd.DataFrame:
    """Return a Pandas dataframe corresponding to the schema of a local URI of a parquet file.

    The returned dataframe has the columns: column, pa_dtype
    """
    import pyarrow.parquet

    # Ref: https://stackoverflow.com/a/64288036/
    schema = pyarrow.parquet.read_schema(path, memory_map=True)
    schema = pd.DataFrame(({"column": name, "pa_dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types)))
    schema = schema.reindex(columns=["column", "pa_dtype"], fill_value=pd.NA)  # Ensures columns in case the parquet file has an empty dataframe.
    return schema
    
    
def cell_reader_factory(path, reader_kwargs={}) -> GDFReader:
    if path.endswith('.geojson'):
        return GeojsonCellReader(**reader_kwargs)
    elif path.endswith('.parquet'):
        schema = read_parquet_schema_df(path)
        if 'geometry' in schema['column'].values:
            return GDFParquetCellReader(**reader_kwargs)
        else:
            return XeniumParquetCellReader(**reader_kwargs)
    else:
        ext = path.split('.')[-1]
        raise ValueError(f'Unknown file extension {ext} for a cell segmentation file, needs to be .geojson or .parquet')
    
    
def read_gdf(path, reader_kwargs={}) -> gpd.GeoDataFrame:
    return cell_reader_factory(path, reader_kwargs).read_gdf(path)
