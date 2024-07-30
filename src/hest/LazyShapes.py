import geopandas as gpd
import pandas as pd
from shapely import Polygon

from hest.io.seg_readers import read_gdf
from hest.utils import verify_paths


class LazyShapes:
    
    path: str = None
    
    def __init__(self, path: str, name: str, coordinate_system: str):
        verify_paths([path])
        self.path = path
        self.name = name
        self.coordinate_system = coordinate_system
        self._shapes = None
        
    def compute(self) -> None:
        if self._shapes is None:
            self._shapes = read_gdf(self.path)
            
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
        shapes.append(Polygon(tissue[:, 0, :]))
        tissue_ids.append(i)
        types.append('tissue')
        holes = contours_holes[i]
        if len(holes) > 0:
            for hole in holes:
                shapes.append(Polygon(hole[:, 0, :]))
                tissue_ids.append(i)
                types.append('hole')
                
    df = pd.DataFrame(tissue_ids, columns=['tissue_id'])
    df['hole'] = types
    df['hole'] = df['hole'] == 'hole'
            
    return gpd.GeoDataFrame(df, geometry=shapes)
        