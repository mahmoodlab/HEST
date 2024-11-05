import geopandas as gpd
import pandas as pd
from shapely import Polygon

from hest.io.seg_readers import GDFReader, read_gdf
from hest.utils import verify_paths


class LazyShapes:
    
    path: str = None
    
    def __init__(self, path: str, name: str, coordinate_system: str, reader: GDFReader=None, reader_kwargs = {}):
        verify_paths([path])
        self.path = path
        self.name = name
        self.coordinate_system = coordinate_system
        self._shapes = None
        self.reader_kwargs = reader_kwargs
        self.reader = reader
        
    def compute(self) -> None:
        if self._shapes is None:
            if self.reader is None:
                self._shapes = read_gdf(self.path, self.reader_kwargs)
            else:
                self._shapes = self.reader(**self.reader_kwargs).read_gdf(self.path)
                
            
    @property
    def shapes(self) -> gpd.GeoDataFrame:
        if self._shapes is None:
            self.compute()

        return self._shapes
    
    def __repr__(self) -> str:
        sup_rep = super().__repr__()
        
        loaded_rep = 'loaded' if self._shapes is not None else 'not loaded'
        
        rep = f"""name: {self.name}, coord-system: {self.coordinate_system}, <{loaded_rep}>"""
        return rep
    

def convert_old_to_gpd(contours_holes, contours_tissue) -> gpd.GeoDataFrame:
    assert len(contours_holes) == len(contours_tissue)
    
    shapes = []
    tissue_ids = []
    types = []
    for i in range(len(contours_holes)):
        tissue = contours_tissue[i]
        tissue_ids.append(i)
        holes = contours_holes[i] if len(contours_holes[i]) > 0 else None
        shapes.append(Polygon(tissue[:, 0, :]), holes=holes)
            
    df = pd.DataFrame(tissue_ids, columns=['tissue_id'])
            
    return gpd.GeoDataFrame(df, geometry=shapes)
        

def old_geojson_to_new(gdf):
    polygons = []
    keys = []
    for key, group in gdf.groupby('tissue_id'):
        holes = []
        for row in group.values:
            if row[2]:
                holes.append([coord for coord in row[0].exterior.coords])
            else:
                exterior = [coord for coord in row[0].exterior.coords]
        polygons.append(Polygon(exterior, holes))
        keys.append(key)
    
    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf['tissue_id'] = keys
    return gdf