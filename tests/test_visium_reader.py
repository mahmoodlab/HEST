import unittest
from os.path import join as _j

import scanpy as sc

from hest.readers import VisiumReader
from hest.utils import get_path_relative


class TestHESTData(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.cur_dir = get_path_relative(__file__, '')
        cur_dir = self.cur_dir
        self.output_dir = _j(cur_dir, 'output_tests')
        

    def test_visium_reader(self):
        cur_dir = self.cur_dir
        fullres_img_path = _j(cur_dir, './assets/SPA154.tif')
        adata = sc.read_h5ad(_j(cur_dir, './assets/SPA154.h5ad'))       
        
        
        st = VisiumReader().read(
            fullres_img_path, # path to a full res image
            bc_matric_path, # path to filtered_feature_bc_matrix.h5
            spatial_coord_path=spatial_coord_path # path to a space ranger spatial/ folder containing either a tissue_positions.csv or tissue_position_list.csv
        )

        # if no spatial folder is provided, but you have an alignment file
        st = VisiumReader().read(
            fullres_img_path, # path to a full res image
            bc_matric_path, # path to filtered_feature_bc_matrix.h5
            alignment_file_path=alignment_file_path # path to a .json alignment file
        )

        # if both the alignment file and the spatial folder are missing, attempt autoalignment
        st = VisiumReader().read(
            fullres_img_path, # path to a full res image
            bc_matric_path, # path to filtered_feature_bc_matrix.h5
        )

if __name__ == '__main__':
    unittest.main()
