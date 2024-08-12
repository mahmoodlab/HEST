import json
import warnings
from abc import abstractmethod

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Point, Polygon
from tqdm import tqdm


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
    

class XeniumParquetCellReader(GDFReader):
    
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
    

def write_geojson(gdf: gpd.GeoDataFrame, path: str, category_key: str, extra_prop=False, uniform_prop=True, index_key: str=None) -> None:
        
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
    
    
def cell_reader_factory(path) -> GDFReader:
    if path.endswith('.geojson'):
        return GeojsonCellReader()
    elif path.endswith('.parquet'):
        schema = read_parquet_schema_df(path)
        if 'geometry' in schema['column'].values:
            return GDFParquetCellReader()
        else:
            return XeniumParquetCellReader()
    else:
        ext = path.split('.')[-1]
        raise ValueError(f'Unknown file extension {ext} for a cell segmentation file, needs to be .geojson or .parquet')
    
    
def read_gdf(path) -> gpd.GeoDataFrame:
    return cell_reader_factory(path).read_gdf(path)