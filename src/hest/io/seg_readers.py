import json
from abc import abstractmethod
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon, Point
from tqdm import tqdm

from hest.utils import align_xenium_df, df_morph_um_to_pxl


class CellReader:
    @abstractmethod
    def read_gdf(self, path) -> gpd.GeoDataFrame:
        pass
    

class XeniumParquetCellReader(CellReader):
    
    def read_gdf(self, path) -> gpd.GeoDataFrame:    
        
        df = pd.read_parquet(path)
        
        df['xy'] = list(zip(df['vertex_x'], df['vertex_y']))
        df = df.drop(['vertex_x', 'vertex_y'], axis=1)
        
        df = df.groupby('cell_id').agg({
            'xy': Polygon
        }).reset_index()
        
        gdf = gpd.GeoDataFrame(df, geometry=df['xy'])
        gdf = gdf.drop(['xy'], axis=1)
        return gdf


class GDFParquetCellReader(CellReader):
    
    def read_gdf(self, path) -> gpd.GeoDataFrame:
        return gpd.read_parquet(path)

class GeojsonCellReader(CellReader):

    def read(self, path) -> pd.DataFrame:
        j = 0
        with open(path) as f:
            ls = json.load(f)
            sub_dfs = []
            for x in ls:
                acc = []
                coords_ls = []
                for i in tqdm(range(len(x['geometry']['coordinates']))):
                    coords = np.array(x['geometry']['coordinates'][i][0])
                    ids = np.array([j for _ in range(coords.shape[0])])
                    acc.append(ids)
                    coords_ls.append(coords)
                    j += 1
                acc = np.array([item for sublist in acc for item in sublist])
                coords = np.vstack(coords_ls)
                name = x['properties']['classification']['name']
                sub_dfs.append(pd.DataFrame(np.column_stack((acc, coords, [name for _ in range(coords.shape[0])])), columns=['cell_id', 'x', 'y', 'class']))
            df = pd.concat(sub_dfs, ignore_index=True)
            
        return df
    
    def _process(self, x, extra_props):
        from shapely.geometry.polygon import Polygon, Point
        
        geom_type = x['geometry']['type']
        if geom_type == 'MultiPoint':
            coords = [Point(x['geometry']['coordinates'][i]) for i in range(len(x['geometry']['coordinates']))]
        elif geom_type == 'MultiPolygon':
            coords = [Polygon(x['geometry']['coordinates'][i][0]) for i in range(len(x['geometry']['coordinates']))]
        else:
            raise ValueError("Doesn't recognize type {geom_type}, must be either MultiPoint or MultiPolygon")
        
        name = x['properties']['classification']['name']
        
        gdf = gpd.GeoDataFrame(geometry=coords)
        gdf['class'] = [name for _ in range(len(gdf))]
        
        extra_props = [k for k in x['properties'].keys() if k not in ['objectType', 'classification']]
        for prop in extra_props:
            val = x['properties'][prop]
            gdf[prop] = [val for _ in range(len(gdf))]
            
        return gdf
    
    
    def read_gdf(self, path, class_name=None, extra_props=False) -> gpd.GeoDataFrame:
        with open(path) as f:
            ls = json.load(f)
            
            sub_gdfs = []
            for x in tqdm(ls):
                sub_gdfs.append(self._process(x, extra_props))
    
            gdf = gpd.GeoDataFrame(pd.concat(sub_gdfs, ignore_index=True))
            gdf['cell_id'] = np.arange(len(gdf))
            
            
            
        return gdf
    
    
class XeniumPolygonCellReader(CellReader):

    def read_gdf(self, path, pixel_size_morph, alignment_path=None) -> gpd.GeoDataFrame:
        df = pd.read_parquet(path)
        
        x_key = 'vertex_x'
        y_key = 'vertex_y'
        
        if alignment_path is not None:  
            df, _, _ = align_xenium_df(alignment_path, pixel_size_morph, df, x_key=x_key, y_key=y_key)
        aligned_nuclei_df = df_morph_um_to_pxl(df, x_key, y_key, pixel_size_morph)
        
        
        df = aligned_nuclei_df
            
        
        df['vertex_x'] = df['vertex_x'].astype(float).round(decimals=2)
        df['vertex_y'] = df['vertex_y'].astype(float).round(decimals=2)
        
        df['combined'] = df[['vertex_x', 'vertex_y']].values.tolist()
        df = df[['cell_id', 'combined']]
        
        #df['cell_id'], _ = pd.factorize(df['cell_id'])
        aggr_df = df.groupby('cell_id').agg({
            'combined': Polygon
            }
        )
        
        gdf = gpd.GeoDataFrame(aggr_df, geometry='combined')

        return gdf
    
    
class XeniumPointCellReader(CellReader):

    def read_gdf(self, path, pixel_size_morph, alignment_path=None) -> gpd.GeoDataFrame:
        df = pd.read_parquet(path)
        
        x_key = 'x_centroid'
        y_key = 'y_centroid'
        
        if alignment_path is not None:  
            df, _, _ = align_xenium_df(alignment_path, pixel_size_morph, df, x_key=x_key, y_key=y_key)
        aligned_nuclei_df = df_morph_um_to_pxl(df, x_key, y_key, pixel_size_morph)
        
        coords = aligned_nuclei_df.copy()
        coords.index = df['cell_id']
        points = gpd.points_from_xy(coords['x_centroid'], coords['y_centroid'])
        
        gdf = gpd.GeoDataFrame(df, geometry=points)
        return gdf
    
    
class CellWriter:
    
    @abstractmethod
    def write(gdf: gpd.GeoDataFrame, path: str) -> None:
        pass
    
def write_geojson(gdf: gpd.GeoDataFrame, path: str, category_key: str, extra_prop=False, uniform_prop=True) -> None:
        
    if isinstance(gdf.geometry.iloc[0], Point):
        geometry = 'MultiPoint'
    elif isinstance(gdf.geometry.iloc[0], Polygon):
        geometry = 'MultiPolygon'
    else:
        raise ValueError(f"gdf.geometry[0] must be of type Point or Polygon, got {type(gdf.geometry.iloc[0])}")
    
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
            for col in [c for c in gdf.columns if c not in [category_key, 'geometry']]:
                if uniform_prop:
                    unique = np.unique(slice[col])
                    if len(unique) != 1:
                        warnings.warn(f"extra property {col} is not uniform for group {group}, found {unique}")
                props[col] = slice[col].iloc[0]
            
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
    colors = plt.get_cmap('hsv', len(names))
    color_dict = {}
    for i in range(len(names)):
        rgb = colors(i)[:3]
        rgb = [int(255 * c) for c in rgb]
        color_dict[names[i]] = rgb
    return color_dict
    
    
def cell_reader_factory(path) -> CellReader:
    if path.endswith('.geojson'):
        return GeojsonCellReader()
    elif path.endswith('.parquet'):
        return XeniumParquetCellReader()
    else:
        ext = path.split('.')[-1]
        raise ValueError(f'Unknown file extension {ext} for a cell segmentation file, needs to be .geojson or .parquet')
    
    
def read_gdf(path) -> gpd.GeoDataFrame:
    return cell_reader_factory(path).read_gdf(path)