import os
import unittest
from os.path import join as _j
from typing import List

import anndata as ad
import openslide
import pyvips
import scanpy as sc

import hest
from hest.autoalign import autoalign_visium
from hest.readers import VisiumReader
from hest.utils import get_path_relative, load_image

try:
    from cucim import CuImage
except ImportError:
    CuImage = None
    print("CuImage is not available. Ensure you have a GPU and cucim installed to use GPU acceleration.")

from hest import HESTData, read_HESTData
from hest.utils import get_path_relative
from hest.wsi import WSI, wsi_factory


class TestHESTReader(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.cur_dir = get_path_relative(__file__, '')
        cur_dir = self.cur_dir
        self.output_dir = _j(cur_dir, 'output_tests', 'reader_tests')
        os.makedirs(self.output_dir, exist_ok=True)
        

    def test_visium_reader_img_matrix_spatial(self):
        cur_dir = self.cur_dir
        fullres_img_path = _j(cur_dir, './assets/WSA_LngSP9258463.jpg')
        bc_matrix_path = _j(cur_dir, './assets/filtered_feature_bc_matrix.h5')
        spatial_coord_path = _j(cur_dir, './assets/spatial')
        
        
        st = VisiumReader().read(
            fullres_img_path, # path to a full res image
            bc_matrix_path, # path to filtered_feature_bc_matrix.h5
            spatial_coord_path=spatial_coord_path # path to a space ranger spatial/ folder containing either a tissue_positions.csv or tissue_position_list.csv
        )
        os.makedirs(_j(self.output_dir, 'img+filtered_matrix+spatial'), exist_ok=True)
        
        st.save(_j(self.output_dir, 'img+filtered_matrix+spatial'), pyramidal=True)
        st.save_spatial_plot(_j(self.output_dir, 'img+filtered_matrix+spatial'), self.output_dir)
        
        
        st.dump_patches(
            self.output_dir,
            'demo',
            target_patch_size=224,
            target_pixel_size=0.5
        )


    def test_visium_reader_img_matrix(self):
        cur_dir = self.cur_dir
        fullres_img_path = _j(cur_dir, './assets/WSA_LngSP9258463.jpg')
        bc_matrix_path = _j(cur_dir, './assets/filtered_feature_bc_matrix.h5')

        # if both the alignment file and the spatial folder are missing, attempt autoalignment
        st = VisiumReader().read(
            fullres_img_path, # path to a full res image
            bc_matrix_path, # path to filtered_feature_bc_matrix.h5
        )
        
        os.makedirs(_j(self.output_dir, 'img+filtered_matrix'), exist_ok=True)
        st.save(_j(self.output_dir, 'img+filtered_matrix'), pyramidal=True)
        st.save_spatial_plot(_j(self.output_dir, 'img+filtered_matrix'), self.output_dir)
        
        st.dump_patches(
            self.output_dir,
            'demo',
            target_patch_size=224,
            target_pixel_size=0.5
        )
        
        print(st)
        
        
    def test_autoalign_to_file(self):
        fullres_img_path = _j(self.cur_dir, './assets/WSA_LngSP9258463.jpg')
        
        fullres_img, _ = load_image(fullres_img_path)
        
        os.makedirs(_j(self.output_dir, 'img+filtered_matrix'), exist_ok=True)
        autoalign_visium(fullres_img, _j(self.output_dir, 'img+filtered_matrix'))
        

class TestHESTData(unittest.TestCase):
   
    @classmethod
    def setUpClass(self):
        self.cur_dir = get_path_relative(__file__, '')
        cur_dir = self.cur_dir
        self.output_dir = _j(cur_dir, 'output_tests/hestdata_tests')
        os.makedirs(self.output_dir, exist_ok=True)
        
        id_list = ['SPA154', 'TENX96', 'TENX131']
        
        self.sts = hest.load_hest('hest_data', id_list)

        
    def test_tissue_seg(self):
        for idx, st in enumerate(self.sts):
            with self.subTest(st_object=idx):
                #st.segment_tissue(method='deep')
                #st.save_tissue_seg_jpg(self.output_dir, name=f'deep_{idx}')
                #st.save_tissue_seg_pkl(self.output_dir, name=f'deep_{idx}')
                #st.save_tissue_vis(self.output_dir, name=f'deep_{idx}')
                
                st.segment_tissue(method='otsu')
                st.save_tissue_contours(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_seg_jpg(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_seg_pkl(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_vis(self.output_dir, name=f'otsu_{idx}')

    #def test_patching(self):
    #    for idx, st in enumerate(self.sts):
    #        with self.subTest(st_object=idx):
    #            name = ''
    #            name += st.meta['id']
    #            st.dump_patches(self.output_dir, name=name)

    #def test_wsi(self):
    #    for idx, st in enumerate(self.sts):
    #        with self.subTest(st_object=idx):
    #            os.makedirs(_j(self.output_dir, f'test_save_{idx}'), exist_ok=True)
    #            st.meta['pixel_size_um_embedded'] = st.pixel_size / 1.5
    #            st.meta['pixel_size_um_estimated'] = st.pixel_size
    #            st.save(_j(self.output_dir, f'test_save_{idx}'), save_img=True, plot_pxl_size=True)


if __name__ == '__main__':
    #TestHESTReader()
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHESTData)
    unittest.TextTestRunner(verbosity=2).run(suite)
