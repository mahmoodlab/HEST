from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from typing import Optional, Union
import warnings
from abc import abstractmethod

import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
from shapely.geometry.polygon import Point, Polygon
from tqdm import tqdm

from hest.utils import align_xenium_df, get_n_threads, is_dask_gdf, read_parquet_dask


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
    """ Lazily read shapes such that read_gdf is called at compute time """

    @abstractmethod
    def read_gdf(self, path: str) -> Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]:
        """ Read shapes

        Args:
            path (str): path to shapes

        Returns:
            Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]: shapes
        """
        pass
   
    
def groupby_shape(df, col, n_threads=0, col_shape='xy'):
    block = df[[col, col_shape]]
    
    groups = defaultdict(lambda: [])
    [groups[row[0]].append(row[1]) for row in block.values]
    key, g = zip(*[
        (key, Polygon(value)) 
        for key, value in groups.items() 
        if len(value) >= 4 or print(f"Warning: key {key} has less than 4 points ({len(value)}), skipping")
    ])

    key = np.array(key)
    g = np.array(g)

    concat = np.column_stack((key, g))

    gdf = gpd.GeoDataFrame(geometry=concat[:, 1])
    gdf.index = concat[:, 0]
    
    import gc
    gc.collect()
    
    return gdf

class XeniumParquetCellReader(GDFReader):
    """ Xenium parquet shape reader """
    
    def __init__(
        self, 
        pixel_size_morph: Optional[float]=None, 
        alignment_matrix=None, 
        use_dask=False
    ):
        """ Xenium parquet shape reader

        Args:
            pixel_size_morph (Optional[float], optional): pixel size of DAPI in um/px. Defaults to None.
            alignment_matrix (np.ndarray, optional): optional alignment matrix. Defaults to None.
            use_dask (bool, optional): whenever to load as a dask geodataframe. Defaults to False.
        """

        self.pixel_size_morph = pixel_size_morph
        self.alignment_matrix = alignment_matrix
        self.use_dask = use_dask
    
    def read_gdf(self, path, n_workers=0) -> Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]:
        
        if self.use_dask:
            df = read_parquet_dask(path, nb_partitions=30)
        else:
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

        if self.use_dask:
            def create_xy_column(pdf):
                pdf['xy'] = list(zip(pdf['vertex_x'], pdf['vertex_y']))
                return pdf

            df = df.map_partitions(create_xy_column)
        else:
            df['xy'] = list(zip(df['vertex_x'], df['vertex_y']))
        df = df.drop(['vertex_x', 'vertex_y'], axis=1)
        
        if self.use_dask:
            import dask_geopandas
            from geopandas.array import GeometryDtype
            
            gdf = dask_geopandas.from_dask_dataframe(df)
            gdf_template = gpd.GeoDataFrame(
                {'geometry': gpd.GeoSeries(dtype=GeometryDtype())},
                crs="EPSG:4326",
                index=pd.Index([], dtype="string", name="index"),
            )
            gdf = gdf.map_partitions(groupby_shape, 'cell_id', meta=gdf_template)
        else:
            gdf = groupby_shape(df, 'cell_id')
        return gdf


class GDFParquetCellReader(GDFReader):
    """ Geopandas parquet shape reader """
    
    def __init__(self, use_dask=False, **kwargs):
        """ Geopandas parquet shape reader

        Args:
            use_dask (bool, optional): whenever to load as a dask geodataframe. Defaults to False.
        """
        self.use_dask = use_dask
    
    
    def read_gdf(self, path) -> Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]:
        if self.use_dask:
            import dask_geopandas as dgpd
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(path)
            total_row_groups = parquet_file.num_row_groups
            row_groups_per_partition = max(1, total_row_groups // 30)
            return dgpd.read_parquet(path, split_row_groups=row_groups_per_partition)
        else:
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
    

class XeniumTranscriptsReader(GDFReader):
    """ Xenium transcript shape reader """
    
    def __init__(self, pixel_size_morph: float, use_dask=False, **kwargs):
        """ Xenium transcript shape reader

        Args:
            pixel_size_morph (float): pixel size of DAPI in um/px
            use_dask (bool, optional): whenever to load as a dask geodataframe. Defaults to False.
        """
        self.use_dask = use_dask
        self.pixel_size_morph = pixel_size_morph
    
    
    def read_gdf(self, path) -> Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]:
        if self.use_dask:
            import dask_geopandas as dgpd
            transcripts_df = read_parquet_dask(path, nb_partitions=30)
            transcripts_df['x_location'] = transcripts_df['x_location'] / self.pixel_size_morph
            transcripts_df['y_location'] = transcripts_df['y_location'] / self.pixel_size_morph
            transcripts_df['geometry'] = dgpd.points_from_xy(transcripts_df, 'x_location', 'y_location')
            transcripts_gdf = dgpd.from_dask_dataframe(transcripts_df)
        else:
            transcripts_gdf = gpd.GeoDataFrame(path, geometry=gpd.points_from_xy(
                transcripts_df['x_location'] / self.pixel_size_morph, 
                transcripts_df['y_location'] / self.pixel_size_morph))
        return transcripts_gdf
    
    
class HESTXeniumTranscriptsReader(GDFReader):
    """ HEST Xenium transcript reader """
    
    def __init__(self, use_dask=False, **kwargs):
        """ HEST Xenium transcript reader

        Args:
            use_dask (bool, optional): whenever to load as a dask geodataframe. Defaults to False.
        """
        self.use_dask = use_dask
    
    
    def read_gdf(self, path) -> Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]:
        if self.use_dask:
            import dask_geopandas as dgpd
            transcripts_df = read_parquet_dask(path, nb_partitions=30)
            transcripts_df['geometry'] = dgpd.points_from_xy(transcripts_df, 'dapi_x', 'dapi_y')
            transcripts_gdf = dgpd.from_dask_dataframe(transcripts_df)
        else:
            transcripts_gdf = gpd.GeoDataFrame(path, geometry=gpd.points_from_xy(
                transcripts_df['dapi_x'], 
                transcripts_df['dapi_y']))
        return transcripts_gdf
    

def _write_geojson(
    gdf: gpd.GeoDataFrame, 
    path: str, 
    geometry=None,
    partition_info=None,
):
    colors = generate_colors(['all', 'test'])
    shapes = gdf.geometry
    
    if partition_info is not None:
        p_index = partition_info['number']
    else:
        p_index = -1
        
    properties = {
        "objectType": "detection",
        "classification": {
            "name": str('group'),
            "color": colors['test']
        }
    }
    
    if isinstance(gdf.geometry.iloc[0], Point):
        shapes = [[point.x, point.y] for point in shapes]
    elif isinstance(gdf.geometry.iloc[0], Polygon):
        shapes = [[[[x, y] for x, y in polygon.exterior.coords]] for polygon in shapes]
    cells = [
        {
        'type': 'Feature',
        'geometry': {
            'type': geometry,
            'coordinates': shape
        },
        "properties": properties
    } for shape in shapes
    ]
    
    partition_str = str(p_index) if p_index >= 0 else ''
    path = path if partition_str == '' else os.path.join(path.removesuffix('.geojson'), f'part.{partition_str}.geojson')
    
    with open(path, 'w') as f:
        json.dump({
            "type": "FeatureCollection",
            "features": cells}
        , f)
    

def write_geojson(
    gdf: Union[gpd.GeoDataFrame, dgpd.GeoDataFrame], 
    path: str, 
) -> None:
    """ Write a (dask) geodataframe in optimized QuPath geojson detection format.

    Args:
        gdf (Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]): _description_
        path (str): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    if not path.endswith('.geojson'):
        raise ValueError(f"path must end in .geojson")
    
    use_dask = is_dask_gdf(gdf)
    
    first_geom = gdf.geometry.head(1).values[0]
    g_type = first_geom.geom_type
        
    if g_type == 'Point':
        geometry = 'MultiPoint'
    elif g_type == 'Polygon':
        geometry = 'Polygon'
    else:
        raise ValueError(
            f"gdf geometry must be Point or Polygon, got {g_type}"
        )
    

    if use_dask:
        meta = gdf.head(0).copy()

        from geopandas.array import GeometryDtype
        meta['geometry'] = gpd.GeoSeries(dtype=GeometryDtype())

        meta = meta.set_geometry('geometry').set_crs("EPSG:4326")
        os.makedirs(path.removesuffix('.geojson'), exist_ok=True)
        gdf.map_partitions(_write_geojson, path, geometry,
                           meta=meta).compute()
    else:
        _write_geojson(gdf, path, geometry)
    
    
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
