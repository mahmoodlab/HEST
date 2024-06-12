import os
import unittest
from os.path import join as _j

import anndata as ad
import openslide
import pyvips
import scanpy as sc
try:
    from cucim import CuImage
except ImportError:
    CuImage = None
    print("CuImage is not available. Ensure you have a GPU and cucim installed to use GPU acceleration.")

from hest import HESTData
from hest.utils import get_path_relative
from hest.wsi import WSI


class TestHESTData(unittest.TestCase):
   
    @classmethod
    def setUpClass(self):
        self.cur_dir = get_path_relative(__file__, '')
        cur_dir = self.cur_dir
        self.output_dir = _j(cur_dir, 'output_tests')
        
        # Create an instance of HESTData
        adata = sc.read_h5ad(_j(cur_dir, './assets/SPA154.h5ad'))
        pixel_size = 0.9206
        
        self.st_objects = []
                    
        
        if CuImage is not None:
            img = CuImage(_j(cur_dir, './assets/SPA154.tif'))
            self.st_objects.append(HESTData(adata, img, pixel_size))
        
        img = openslide.OpenSlide(_j(cur_dir, './assets/SPA154.tif'))
        self.st_objects.append(HESTData(adata, img, pixel_size))
        
        if CuImage is not None:
            img = WSI(CuImage(_j(cur_dir, './assets/SPA154.tif'))).numpy()
            self.st_objects.append(HESTData(adata, img, pixel_size))
        else:
            img = WSI(openslide.OpenSlide(_j(cur_dir, './assets/SPA154.tif'))).numpy()
            self.st_objects.append(HESTData(adata, img, pixel_size))    
        
    def test_tissue_seg(self):
        for idx, st in enumerate(self.st_objects):
            with self.subTest(st_object=idx):
                st.compute_mask(method='deep')
                st.save_tissue_seg_jpg(self.output_dir, name=f'deep_{idx}')
                st.save_tissue_seg_pkl(self.output_dir, name=f'deep_{idx}')
                
                st.compute_mask(method='otsu')
                st.save_tissue_seg_jpg(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_seg_pkl(self.output_dir, name=f'otsu_{idx}')

    def test_patching(self):
        for idx, st in enumerate(self.st_objects):
            with self.subTest(st_object=idx):
                st.dump_patches(self.output_dir)

    def test_wsi(self):
        for idx, st in enumerate(self.st_objects):
            with self.subTest(st_object=idx):
                os.makedirs(_j(self.output_dir, f'test_save_{idx}'), exist_ok=True)
                st.save(_j(self.output_dir, f'test_save_{idx}'), save_img=True)


if __name__ == '__main__':
    unittest.main()
