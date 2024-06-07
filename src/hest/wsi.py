import cv2
import numpy as np
import openslide
from cucim import CuImage
from openslide.deepzoom import DeepZoomGenerator


class WSI:
    def __init__(self, img, patch_size_src=512, overlap=0):
        self.img = img
        
        self.width, self.height = self.get_dimensions()
        
        self.patch_size_src = patch_size_src
        self.overlap = overlap
        cols, rows = self.get_cols_rows()
        

    def get_dimensions(self):
        img = self.img
        if isinstance(img, openslide.OpenSlide):
            width, height = img.dimensions
        elif isinstance(img, np.ndarray):
            width, height = img.shape[1], img.shape[0]
        elif isinstance(img, CuImage):
            width, height = img.resolutions['level_dimensions'][0]
        return width, height
    
    def get_cols_rows(self):
        img = self.img
        if isinstance(img, openslide.OpenSlide):
            self.dz = DeepZoomGenerator(img, self.patch_size_src, self.overlap)
            cols, rows = self.dz.level_tiles[-1]
            self.nb_levels = len(self.dz.level_tiles)
        elif isinstance(img, CuImage) or isinstance(img, np.ndarray):
            cols, rows = round(np.ceil((self.width - self.overlap / 2) / (self.patch_size_src - self.overlap / 2))), round(np.ceil((self.height - self.overlap / 2) / (self.patch_size_src - self.overlap / 2)))
        return cols, rows
    
    def get_tile(self, col, row):
        img = self.img
        if isinstance(img, openslide.OpenSlide):
            raw_tile = self.dz.get_tile(self.nb_levels - 1, (col, row))
            addr = self.dz.get_tile_coordinates(self.nb_levels - 1, (col, row))
            pxl_x, pxl_y = addr[0]
        elif isinstance(img, CuImage):
            x_begin = round(col * (self.patch_size_src - self.overlap))
            y_begin = round(row * (self.patch_size_src - self.overlap))
            raw_tile = self.img.read_region(location=(x_begin, y_begin), level=0, size=(self.patch_size_src + self.overlap, self.patch_size_src + self.overlap))
            pxl_x = x_begin
            pxl_y = y_begin
        elif isinstance(img, np.ndarray):
            x_begin = round(col * (self.patch_size_src - self.overlap))
            x_end = min(x_begin + self.patch_size_src + self.overlap, self.width)
            y_begin = round(row * (self.patch_size_src - self.overlap))
            y_end = min(y_begin + self.patch_size_src + self.overlap, self.height)
            tmp_tile = np.zeros((self.patch_size_src, self.patch_size_src, 3), dtype=np.uint8)
            tmp_tile[:y_end-y_begin, :x_end-x_begin] += img[y_begin:y_end, x_begin:x_end]
            pxl_x, pxl_y = x_begin, y_begin
            raw_tile = tmp_tile
            
        tile = np.array(raw_tile)
        assert pxl_x < self.width and pxl_y < self.height
        return tile, pxl_x, pxl_y
    
    def get_thumbnail(self, width, height):
        img = self.img
        if isinstance(img, np.ndarray):
            thumbnail = cv2.resize(img, (width, height))
        elif isinstance(img, CuImage):
            downsample = self.width / width
            downsamples = img.resolutions['level_downsamples']
            closest = 0
            for i in range(len(downsamples)):
                if downsamples[i] > downsample:
                    break
                closest = i
            
            curr_width, curr_height = img.resolutions['level_dimensions'][closest]
            thumbnail = np.array(img.read_region(location=(0, 0), level=closest, size=(curr_width, curr_height)))
            thumbnail = cv2.resize(thumbnail, (width, height))
        elif isinstance(img, openslide.OpenSlide):
            thumbnail =  np.array(img.get_thumbnail((width, height)))
            
        return thumbnail
        