import warnings
from abc import abstractmethod
from typing import Tuple

import cv2
import geopandas as gpd
import numpy as np
import openslide


class CucimWarningSingleton:
    _warned_cucim = False

    @classmethod
    def warn(cls):
        if cls._warned_cucim is False:
            warnings.warn("CuImage is not available. Ensure you have a GPU and cucim installed to use GPU acceleration.")
            cls._warned_cucim = True
        return cls._warned_cucim


def is_cuimage(img):
    try:
        from cucim import CuImage
    except ImportError:
        CuImage = None
        CucimWarningSingleton.warn()
    return CuImage is not None and isinstance(img, CuImage) # type: ignore


class WSI:
    
    def __init__(self, img):
        self.img = img
        
        if not (isinstance(img, openslide.OpenSlide) or isinstance(img, np.ndarray) or is_cuimage(img)) :
            raise ValueError(f"Invalid type for img {type(img)}")
        
        self.width, self.height = self.get_dimensions()
        
    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dimensions(self):
        pass
    
    @abstractmethod
    def read_region(self, location, level, size) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_thumbnail(self, width, height):
        pass
    
    def __repr__(self) -> str:
        width, height = self.get_dimensions()
        
        return f"<width={width}, height={height}, backend={self.__class__.__name__}>"
    
    @abstractmethod
    def create_patcher(self, patch_size_src: int, patch_size_target: int = None, overlap: int = 0, mask: gpd.GeoDataFrame = None, coords_only = False):
        pass
    

def wsi_factory(img) -> WSI:
    try:
        from cucim import CuImage
    except ImportError:
        CuImage = None
        CucimWarningSingleton.warn()
    
    if isinstance(img, WSI):
        return img
    elif isinstance(img, openslide.OpenSlide):
        return OpenSlideWSI(img)
    elif isinstance(img, np.ndarray):
        return NumpyWSI(img)
    elif is_cuimage(img):
        return CuImageWSI(img)
    elif isinstance(img, str):
        if CuImage is not None:
            return CuImageWSI(CuImage(img))
        else:
            warnings.warn("Cucim isn't available, opening the image with OpenSlide (will be slower)")
            return OpenSlideWSI(openslide.OpenSlide(img))
    else:
        raise ValueError(f'type {type(img)} is not supported')

class NumpyWSI(WSI):
    def __init__(self, img: np.ndarray):
        super().__init__(img)
        
    def numpy(self) -> np.ndarray:
        return self.img

    def get_dimensions(self):
        return self.img.shape[1], self.img.shape[0]
    
    def read_region(self, location, level, size) -> np.ndarray:
        img = self.img
        x_start, y_start = location[0], location[1]
        x_size, y_size = size[0], size[1]
        return img[y_start:y_start + y_size, x_start:x_start + x_size]
    
    def get_thumbnail(self, width, height) -> np.ndarray:
        return cv2.resize(self.img, (width, height))
    
    def create_patcher(self, patch_size_src: int, patch_size_target: int = None, overlap: int = 0, mask: gpd.GeoDataFrame = None, coords_only = False):
        return NumpyWSIPatcher(self, patch_size_src, patch_size_target, overlap, mask, coords_only)
    

class OpenSlideWSI(WSI):
    def __init__(self, img: openslide.OpenSlide):
        super().__init__(img)
        
    def numpy(self) -> np.ndarray:
        return self.get_thumbnail(self.width, self.height)

    def get_dimensions(self):
        return self.img.dimensions
    
    def read_region(self, location, level, size) -> np.ndarray:
        return np.array(self.img.read_region(location, level, size))

    def get_thumbnail(self, width, height):
        return np.array(self.img.get_thumbnail((width, height)))
    
    def get_best_level_for_downsample(self, downsample):
        return self.img.get_best_level_for_downsample(downsample)
    
    def level_dimensions(self):
        return self.img.level_dimensions
    
    def level_downsamples(self):
        return self.img.level_downsamples
    
    def create_patcher(self, patch_size_src: int, patch_size_target: int = None, overlap: int = 0, mask: gpd.GeoDataFrame = None, coords_only = False):
        return OpenSlideWSIPatcher(self, patch_size_src, patch_size_target, overlap, mask, coords_only)
    
class CuImageWSI(WSI):
    def __init__(self, img: 'CuImage'):
        super().__init__(img)

    def numpy(self) -> np.ndarray:
        return self.get_thumbnail(self.width, self.height)

    def get_dimensions(self):
        return self.img.resolutions['level_dimensions'][0]
    
    def read_region(self, location, level, size) -> np.ndarray:
        return np.array(self.img.read_region(location=location, level=level, size=size))
    
    def get_thumbnail(self, width, height):
        downsample = self.width / width
        downsamples = self.img.resolutions['level_downsamples']
        closest = 0
        for i in range(len(downsamples)):
            if downsamples[i] > downsample:
                break
            closest = i
        
        curr_width, curr_height = self.img.resolutions['level_dimensions'][closest]
        thumbnail = np.array(self.img.read_region(location=(0, 0), level=closest, size=(curr_width, curr_height)))
        thumbnail = cv2.resize(thumbnail, (width, height))            
            
        return thumbnail
    
    def get_best_level_for_downsample(self, downsample):
        downsamples = self.img.resolutions['level_downsamples']
        last = 0
        for i in range(len(downsamples)):
            down = downsamples[i]
            if downsample < down:
                return last
            last = i
        return last
    
    def level_dimensions(self):
        return self.img.resolutions['level_dimensions']
    
    def level_downsamples(self):
        return self.img.resolutions['level_downsamples']
    
    def create_patcher(self, patch_size_src: int, patch_size_target: int = None, overlap: int = 0, mask: gpd.GeoDataFrame = None, coords_only = False):
        return CuImageWSIPatcher(self, patch_size_src, patch_size_target, overlap, mask, coords_only)
            
        
class WSIPatcher:
    """ Iterator class to handle patching, patch scaling and tissue mask intersection """
    
    def __init__(
        self, 
        wsi: WSI, 
        patch_size: int, 
        patch_size_target: int = None, 
        overlap: int = 0,
        mask: gpd.GeoDataFrame = None,
        coords_only = False
    ):
        """ Initialize patcher, compute number of (masked) rows, columns.

        Args:
            wsi (WSI): wsi to patch
            patch_size (int): patch width/height in pixel on the slide before rescaling
            patch_size_target (int, optional): largest patch size in pixel after rescaling. Defaults to None.
            overlap (int, optional): overlap size in pixel before rescaling. Defaults to 0.
            mask (gpd.GeoDataFrame, optional): geopandas dataframe of Polygons. Defaults to None.
            coords_only (bool, optional): whenever to extract only the coordinates insteaf of coordinates + tile. Default to False.
        """
        self.wsi = wsi
        self.patch_size = patch_size
        self.overlap = overlap
        self.width, self.height = self.wsi.get_dimensions()
        self.patch_size_target = patch_size_target
        self.mask = mask
        self.i = 0
        self.coords_only = coords_only
        
        if patch_size_target is None:
            self.downsample = 1.
        else:
            self.downsample = patch_size / patch_size_target
        
        self.level, self.patch_size_level, self.overlap_level = self._prepare()
        self.cols, self.rows = self._compute_cols_rows()   
        
        col_rows = np.array([
            [col, row] 
            for col in range(self.cols) 
            for row in range(self.rows)
        ])
        
        if self.mask is not None:
            self.valid_patches_nb, self.valid_col_rows = self._compute_masked(col_rows)
        else:
            self.valid_patches_nb, self.valid_col_rows = len(col_rows), col_rows
            
    def _colrow_to_xy(self, col, row):
        """ Convert col row of a tile to its top-left coordinates before rescaling (x, y) """
        x = col * (self.patch_size) if col == 0 else col * (self.patch_size) - self.overlap
        y = row * (self.patch_size) if row == 0 else row * (self.patch_size) - self.overlap
        return (x, y)   
            
    def _compute_masked(self, col_rows) -> None:
        """ Compute tiles which center falls under the tissue """
        
        xy_topleft = np.array([self._colrow_to_xy(xy[0], xy[1]) for xy in col_rows])
        
        # Note: we don't take into account the overlap size we calculating centers
        xy_centers = xy_topleft + self.patch_size_level // 2
        
        union_mask = self.mask.unary_union
        
        points = gpd.points_from_xy(xy_centers)
        valid_mask = gpd.GeoSeries(points).within(union_mask).values
        valid_patches_nb = valid_mask.sum()
        valid_col_rows = col_rows[valid_mask]
        return valid_patches_nb, valid_col_rows
        
    def __len__(self):
        return self.valid_patches_nb
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.valid_patches_nb:
            raise StopIteration
        x = self.__getitem__(self.i)
        self.i += 1
        return x
    
    def __getitem__(self, index):
        if 0 <= index < len(self.valid_col_rows):
            col_row = self.valid_col_rows[index]
            col, row = col_row[0], col_row[1]
            if self.coords_only:
                return self._colrow_to_xy(col, row)
            tile, x, y = self.get_tile(col, row)
            return tile, x, y
        else:
            raise IndexError("Index out of range")
        

    @abstractmethod
    def _prepare(self) -> None:
        pass
    
    def get_cols_rows(self) -> Tuple[int, int]:
        """ Get the number of columns and rows in the associated WSI

        Returns:
            Tuple[int, int]: (nb_columns, nb_rows)
        """
        return self.cols, self.rows
    
    def get_tile(self, col: int, row: int) -> Tuple[np.ndarray, int, int]:
        """ get tile at position (column, row)

        Args:
            col (int): column
            row (int): row

        Returns:
            Tuple[np.ndarray, int, int]: (tile, pixel x of top-left corner (before rescaling), pixel_y of top-left corner (before rescaling))
        """
        x, y = self._colrow_to_xy(col, row)
        raw_tile = self.wsi.read_region(location=(x, y), level=self.level, size=(self.patch_size_level, self.patch_size_level))
        tile = np.array(raw_tile)
        if self.patch_size_target is not None:
            tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))
        assert x < self.width and y < self.height
        return tile, x, y
    
    def _compute_cols_rows(self) -> Tuple[int, int]:
        col = 0
        row = 0
        x, y = self._colrow_to_xy(col, row)
        while x < self.width:
            col += 1
            x, _ = self._colrow_to_xy(col, row)
        cols = col - 1
        while y < self.height:
            row += 1
            _, y = self._colrow_to_xy(col, row)
        rows = row - 1
        return cols, rows 
    
    
class OpenSlideWSIPatcher(WSIPatcher):
    wsi: OpenSlideWSI
    
    def _prepare(self) -> None:
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level
    
class CuImageWSIPatcher(WSIPatcher):
    wsi: CuImageWSI
    
    def _prepare(self) -> None:
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level

class NumpyWSIPatcher(WSIPatcher):
    WSI: NumpyWSI
    
    def _prepare(self) -> None:
        patch_size_level = self.patch_size
        overlap_level = self.overlap
        level = -1
        return level, patch_size_level, overlap_level