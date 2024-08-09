from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import partial
from typing import Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import openslide
from PIL import Image
from shapely import Polygon


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
    def create_patcher(
        self, 
        patch_size: int, 
        src_pixel_size: float, 
        dst_pixel_size: float = None, 
        overlap: int = 0, 
        mask: gpd.GeoDataFrame = None, 
        coords_only = False, 
        custom_coords = None
    ) -> WSIPatcher:
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
        x_end, y_end = x_start + x_size, y_start + y_size
        padding_left = max(0 - x_start, 0)
        padding_top = max(0 - y_start, 0)
        padding_right = max(x_start + x_size - self.width, 0)
        padding_bottom = max(y_start + y_size - self.height, 0)
        x_start, y_start = max(x_start, 0),  max(y_start, 0)
        x_end, y_end = min(x_end, self.width), min(y_end, self.height)
        tile = img[y_start:y_end, x_start:x_end]
        padded_tile = np.pad(tile, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), mode='constant', constant_values=0)
        
        return padded_tile
    
    def get_thumbnail(self, width, height) -> np.ndarray:
        return cv2.resize(self.img, (width, height))
    
    def create_patcher(
        self, 
        patch_size: int, 
        src_pixel_size: float, 
        dst_pixel_size: float = None, 
        overlap: int = 0, 
        mask: gpd.GeoDataFrame = None, 
        coords_only = False, 
        custom_coords = None
    ) -> WSIPatcher:
        return NumpyWSIPatcher(self, patch_size, src_pixel_size, dst_pixel_size, overlap, mask, coords_only, custom_coords)
    

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
    
    def create_patcher(
        self, 
        patch_size: int, 
        src_pixel_size: float, 
        dst_pixel_size: float = None, 
        overlap: int = 0, 
        mask: gpd.GeoDataFrame = None, 
        coords_only = False, 
        custom_coords = None
    ) -> WSIPatcher:
        return OpenSlideWSIPatcher(self, patch_size, src_pixel_size, dst_pixel_size, overlap, mask, coords_only, custom_coords)
    
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
    
    def create_patcher(
        self, 
        patch_size: int, 
        src_pixel_size: float, 
        dst_pixel_size: float = None, 
        overlap: int = 0, 
        mask: gpd.GeoDataFrame = None, 
        coords_only = False, 
        custom_coords = None
    ) -> WSIPatcher:
        return CuImageWSIPatcher(self, patch_size, src_pixel_size, dst_pixel_size, overlap, mask, coords_only, custom_coords)
            
        
class WSIPatcher:
    """ Iterator class to handle patching, patch scaling and tissue mask intersection """
    
    def __init__(
        self, 
        wsi: WSI, 
        patch_size: int, 
        src_pixel_size: float,
        dst_pixel_size: float = None,
        overlap: int = 0,
        mask: gpd.GeoDataFrame = None,
        coords_only = False,
        custom_coords = None,
        threshold = 0.15
    ):
        """ Initialize patcher, compute number of (masked) rows, columns.

        Args:
            wsi (WSI): wsi to patch
            patch_size (int): patch width/height in pixel on the slide after rescaling
            src_pixel_size (float, optional): pixel size in um/px of the slide before rescaling. Defaults to None.
            dst_pixel_size (float, optional): pixel size in um/px of the slide after rescaling. Defaults to None.
            overlap (int, optional): overlap size in pixel before rescaling. Defaults to 0.
            mask (gpd.GeoDataFrame, optional): geopandas dataframe of Polygons. Defaults to None.
            coords_only (bool, optional): whenever to extract only the coordinates insteaf of coordinates + tile. Default to False.
            threshold (float, optional): minimum proportion of the patch under tissue to be kept.
                This argument is ignored if mask=None, passing threshold=0 will be faster
        """
        self.wsi = wsi
        self.overlap = overlap
        self.width, self.height = self.wsi.get_dimensions()
        self.patch_size_target = patch_size
        self.mask = mask
        self.i = 0
        self.coords_only = coords_only
        self.custom_coords = custom_coords
        
        if dst_pixel_size is None:
            self.downsample = 1.
        else:
            self.downsample = dst_pixel_size / src_pixel_size
            
        self.patch_size_src = round(patch_size * self.downsample)
        
        self.level, self.patch_size_level, self.overlap_level = self._prepare()  
        
        if custom_coords is None: 
            self.cols, self.rows = self._compute_cols_rows()
            
            col_rows = np.array([
                [col, row] 
                for col in range(self.cols) 
                for row in range(self.rows)
            ])
            coords = np.array([self._colrow_to_xy(xy[0], xy[1]) for xy in col_rows])
        else:
            if round(custom_coords[0][0]) != custom_coords[0][0]:
                raise ValueError("custom_coords must be a (N, 2) array of int")
            coords = custom_coords
        if self.mask is not None:
            self.valid_patches_nb, self.valid_coords = self._compute_masked(coords, threshold)
        else:
            self.valid_patches_nb, self.valid_coords = len(coords), coords
            
    def _colrow_to_xy(self, col, row):
        """ Convert col row of a tile to its top-left coordinates before rescaling (x, y) """
        x = col * (self.patch_size_src) - self.overlap * np.clip(col - 1, 0, None)
        y = row * (self.patch_size_src) - self.overlap * np.clip(row - 1, 0, None)
        return (x, y)   
            
    def _compute_masked(self, coords, threshold) -> None:
        """ Compute tiles which center falls under the tissue """
        
		# Filter coordinates by bounding boxes of mask polygons
        bounding_boxes = self.mask.geometry.bounds
        valid_coords = []
        for _, bbox in bounding_boxes.iterrows():
            bbox_coords = coords[
                (coords[:, 0] >= bbox['minx'] - self.patch_size_src) & (coords[:, 0] <= bbox['maxx'] + self.patch_size_src) & 
                (coords[:, 1] >= bbox['miny'] - self.patch_size_src) & (coords[:, 1] <= bbox['maxy'] + self.patch_size_src)
            ]
            valid_coords.append(bbox_coords)

        if len(valid_coords) > 0:
            coords = np.vstack(valid_coords)
            coords = np.unique(coords, axis=0)
        else:
            coords = np.array([])
            
        
        union_mask = self.mask.union_all()

        squares = [
            Polygon([
                (xy[0], xy[1]), 
                (xy[0] + self.patch_size_src, xy[1]), 
                (xy[0] + self.patch_size_src, xy[1] + self.patch_size_src), 
                (xy[0], xy[1] + self.patch_size_src)]) 
            for xy in coords
        ]
        if threshold == 0:
            valid_mask = gpd.GeoSeries(squares).intersects(union_mask).values
        else:
            gdf = gpd.GeoSeries(squares)
            areas = gdf.area
            valid_mask = gdf.intersection(union_mask).area >= threshold * areas

        valid_patches_nb = valid_mask.sum()
        valid_coords = coords[valid_mask]
        return valid_patches_nb, valid_coords
        
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
        if 0 <= index < len(self):
            xy = self.valid_coords[index]
            x, y = xy[0], xy[1]
            if self.coords_only:
                return x, y
            tile, x, y = self.get_tile_xy(x, y)
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
      
    def get_tile_xy(self, x: int, y: int) -> Tuple[np.ndarray, int, int]:
        raw_tile = self.wsi.read_region(location=(x, y), level=self.level, size=(self.patch_size_level, self.patch_size_level))
        tile = np.array(raw_tile)
        if self.patch_size_target is not None:
            tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))
        assert x < self.width and y < self.height
        return tile[:, :, :3], x, y
    
    def get_tile(self, col: int, row: int) -> Tuple[np.ndarray, int, int]:
        """ get tile at position (column, row)

        Args:
            col (int): column
            row (int): row

        Returns:
            Tuple[np.ndarray, int, int]: (tile, pixel x of top-left corner (before rescaling), pixel_y of top-left corner (before rescaling))
        """
        if self.custom_coords is not None:
            raise ValueError("Can't use get_tile as 'custom_coords' was passed to the constructor")
            
        x, y = self._colrow_to_xy(col, row)
        return self.get_tile_xy(x, y)
    
    def _compute_cols_rows(self) -> Tuple[int, int]:
        col = 0
        row = 0
        x, y = self._colrow_to_xy(col, row)
        while x < self.width:
            col += 1
            x, _ = self._colrow_to_xy(col, row)
        cols = col
        while y < self.height:
            row += 1
            _, y = self._colrow_to_xy(col, row)
        rows = row
        return cols, rows 
    
    def save_visualization(self, path, vis_width=1000, dpi=150):
        mask_plot = get_tissue_vis(
            self.wsi,
            self.mask,
            line_color=(0, 255, 0),
            line_thickness=5,
            target_width=vis_width,
            seg_display=True,
        )
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        
        downscale_vis = vis_width / self.width
        
        _, ax = plt.subplots()
        ax.imshow(mask_plot)
        
        patch_rectangles = []
        for xy in self.valid_coords:
            x, y = xy[0], xy[1]
            x, y = x * downscale_vis, y * downscale_vis
        
            patch_rectangles.append(Rectangle((x, y), self.patch_size_src * downscale_vis, self.patch_size_src * downscale_vis))
        
        ax.add_collection(PatchCollection(patch_rectangles, facecolor='none', edgecolor='black', linewidth=0.3))
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches = 'tight')
    
    
class OpenSlideWSIPatcher(WSIPatcher):
    wsi: OpenSlideWSI
    
    def _prepare(self) -> None:
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size_src / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level
    
class CuImageWSIPatcher(WSIPatcher):
    wsi: CuImageWSI
    
    def _prepare(self) -> None:
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size_src / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level

class NumpyWSIPatcher(WSIPatcher):
    WSI: NumpyWSI
    
    def _prepare(self) -> None:
        patch_size_level = self.patch_size_src
        overlap_level = self.overlap
        level = -1
        return level, patch_size_level, overlap_level
    
    
    
def contours_to_img(
    contours: gpd.GeoDataFrame, 
    img: np.ndarray, 
    draw_contours=False, 
    thickness=1, 
    downsample=1.,
    line_color=(0, 255, 0)
) -> np.ndarray:
    draw_cont = partial(cv2.drawContours, contourIdx=-1, thickness=thickness, lineType=cv2.LINE_8)
    draw_cont_fill = partial(cv2.drawContours, contourIdx=-1, thickness=cv2.FILLED)
    
    groups = contours.groupby('tissue_id')
    for _, group in groups:
        
        for _, row in group.iterrows():
            cont = np.array([[round(x * downsample), round(y * downsample)] for x, y in row.geometry.exterior.coords])
            holes = [np.array([[round(x * downsample), round(y * downsample)] for x, y in hole.coords]) for hole in row.geometry.interiors]
        
            draw_cont_fill(image=img, contours=[cont], color=line_color)
        
            for hole in holes:
                draw_cont_fill(image=img, contours=[hole], color=(0, 0, 0))

            if draw_contours:
                draw_cont(image=img, contours=[cont], color=line_color)
    return img


def get_tissue_vis(
            img: Union[np.ndarray, openslide.OpenSlide, CuImage, WSI],
            tissue_contours: gpd.GeoDataFrame,
            line_color=(0, 255, 0),
            line_thickness=5,
            target_width=1000,
            seg_display=True,
    ) -> Image:
    
        wsi = wsi_factory(img)
    
        width, height = wsi.get_dimensions()
        downsample = target_width / width

        top_left = (0,0)
        
        img = wsi.get_thumbnail(round(width * downsample), round(height * downsample))

        if tissue_contours is None:
            return Image.fromarray(img)
        
        tissue_contours = tissue_contours.copy()

        downscaled_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        downscaled_mask = np.expand_dims(downscaled_mask, axis=-1)
        downscaled_mask = downscaled_mask * np.array([0, 0, 0]).astype(np.uint8)

        if tissue_contours is not None and seg_display:
            downscaled_mask = contours_to_img(
                tissue_contours, 
                downscaled_mask, 
                draw_contours=True, 
                thickness=line_thickness, 
                downsample=downsample,
                line_color=line_color
            )

        alpha = 0.4
        img = cv2.addWeighted(img, 1 - alpha, downscaled_mask, alpha, 0)
        img = img.astype(np.uint8)

        return Image.fromarray(img)