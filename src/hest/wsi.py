import warnings
from abc import abstractmethod
from typing import Tuple

import cv2
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator

from hest.utils import warn_cucim


def is_cuimage(img):
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
    

def wsi_factory(img) -> WSI:
    try:
        from cucim import CuImage
    except ImportError:
        CuImage = None
        warn_cucim()
    
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
            
        
class WSIPatcher:
    def __init__(self, wsi: WSI, patch_size_src: int, patch_size_target: int = None):
        self.wsi = wsi
        self.patch_size_src = patch_size_src
        self.overlap = 0
        self.width, self.height = self.wsi.get_dimensions()
        self.patch_size_target = patch_size_target
        self.downsample = patch_size_src / patch_size_target
        
        self._compute_cols_rows()
        
    def _compute_cols_rows(self) -> None:
        img = self.wsi.img
        if isinstance(img, openslide.OpenSlide):
            self.level = self.wsi.get_best_level_for_downsample(self.downsample)
            self.level_dimensions = self.wsi.level_dimensions()[self.level]
            self.level_downsample = self.wsi.level_downsamples()[self.level]
            self.patch_size_level = round(self.patch_size_src / self.level_downsample)
            self.dz = DeepZoomGenerator(img, self.patch_size_level, self.overlap)
            self.nb_levels = len(self.dz.level_tiles)
            self.cols, self.rows = self.dz.level_tiles[self.nb_levels - self.level - 1]
        elif isinstance(img, np.ndarray):
            self.cols, self.rows = round(np.ceil((self.width - self.overlap / 2) / (self.patch_size_src - self.overlap / 2))), round(np.ceil((self.height - self.overlap / 2) / (self.patch_size_src - self.overlap / 2)))
            self.level = -1
            self.level_dimensions = (self.width, self.height)
        elif is_cuimage(img):
            self.level = self.wsi.get_best_level_for_downsample(self.downsample)
            self.level_downsample = self.wsi.level_downsamples()[self.level]
            self.level_dimensions = self.wsi.level_dimensions()[self.level]
            self.patch_size_level = round(self.patch_size_src / self.level_downsample)
            level_width, level_height = self.level_dimensions
            self.cols, self.rows = round(np.ceil((level_width - self.overlap / 2) / (self.patch_size_level - self.overlap / 2))), round(np.ceil((level_height - self.overlap / 2) / (self.patch_size_level - self.overlap / 2)))
    

    def get_cols_rows(self) -> Tuple[int, int]:
        """ Get the number of columns and rows the associated WSI

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
            Tuple[np.ndarray, int, int]: (tile, pixel x of top-left corner, pixel_y of top-left corner)
        """
        img = self.wsi.img
        if isinstance(img, openslide.OpenSlide):
            raw_tile = self.dz.get_tile(self.nb_levels - self.level - 1, (col, row))
            addr = self.dz.get_tile_coordinates(self.nb_levels - self.level - 1, (col, row))
            pxl_x, pxl_y = addr[0]
            if pxl_x == 556 and pxl_y == 556:
                a = 1
        elif isinstance(img, np.ndarray):
            x_begin = round(col * (self.patch_size_src - self.overlap))
            x_end = min(x_begin + self.patch_size_src + self.overlap, self.width)
            y_begin = round(row * (self.patch_size_src - self.overlap))
            y_end = min(y_begin + self.patch_size_src + self.overlap, self.height)
            tmp_tile = np.zeros((self.patch_size_src, self.patch_size_src, 3), dtype=np.uint8)
            tmp_tile[:y_end-y_begin, :x_end-x_begin] += img[y_begin:y_end, x_begin:x_end]
            pxl_x, pxl_y = x_begin, y_begin
            raw_tile = tmp_tile
        elif is_cuimage(img):
            x_begin = round(col * (self.patch_size_src - self.overlap))
            y_begin = round(row * (self.patch_size_src - self.overlap))
            raw_tile = self.wsi.read_region(location=(x_begin, y_begin), level=self.level, size=(self.patch_size_level + self.overlap, self.patch_size_level + self.overlap))
            pxl_x = x_begin
            pxl_y = y_begin
            
        tile = np.array(raw_tile)
        if self.patch_size_target is not None:
            tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))
        assert pxl_x < self.width and pxl_y < self.height
        return tile, pxl_x, pxl_y