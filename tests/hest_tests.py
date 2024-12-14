import os
import unittest
import warnings
from datetime import datetime
from os.path import join as _j

from hestcore.segmentation import get_path_relative
from hestcore.wsi import CucimWarningSingleton

MAX_HEST_IMPORT_S = 2
start_time = datetime.now()
import hest
end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
if elapsed_time > MAX_HEST_IMPORT_S:
    raise ImportError(f"Importing 'hest' took too long ({elapsed_time:.2f} seconds). Maximum allowed time is {MAX_HEST_IMPORT_S} seconds. Please, keep large large imports conditional")

from hest.autoalign import autoalign_visium
from hest.readers import VisiumReader
from hest.HESTData import ensembl_id_to_gene
from hest.utils import load_image


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
        
        from huggingface_hub import login
        
        token = os.getenv('HF_READ_TOKEN_PAUL')
        if token is None:
            download = False
            warnings.warn("Please setup huggingface token 'HF_READ_TOKEN_PAUL'")
        else:
            download = True
            login(token=token)
        download=True
        
        id_list = ['TENX24', 'SPA154']
        
        if download:
            import datasets
            
            local_dir = os.path.join(cur_dir, 'hest_data_test')
            
            ids_to_query = id_list
            list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
            datasets.load_dataset(
                'MahmoodLab/hest', 
                cache_dir=local_dir,
                patterns=list_patterns,
                #download_mode='force_redownload',
                trust_remote_code=True
            )
            
            self.sts = hest.load_hest(local_dir, id_list)
        else:
            self.sts = hest.load_hest('hest_data', id_list)


    #def test_conversion_ensembleID(self):
    #    for idx, st in enumerate(self.sts):
    #        with self.subTest(st_object=idx):
    #            ensembl_id_to_gene(st)

        
    def test_tissue_seg(self):
        for idx, st in enumerate(self.sts):
            with self.subTest(st_object=idx):
                st.segment_tissue(method='deep')
                st.save_tissue_contours(self.output_dir, name=f'deep_{idx}')
                st.save_tissue_seg_jpg(self.output_dir, name=f'deep_{idx}')
                st.save_tissue_seg_pkl(self.output_dir, name=f'deep_{idx}')
                st.save_tissue_vis(self.output_dir, name=f'deep_{idx}')
                
                st.segment_tissue(method='otsu')
                st.save_tissue_contours(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_seg_jpg(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_seg_pkl(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_vis(self.output_dir, name=f'otsu_{idx}')


    def test_spatialdata(self):
        for idx, st in enumerate(self.sts):
            with self.subTest(st_object=idx):
                name = ''
                name += st.meta['id']
                spd = st.to_spatial_data()
                print(spd)


    def test_patching(self):
        """ Save patches as .h5 then load with H5HESTDataset """
        from hestcore.datasets import H5HESTDataset
        from PIL import Image, ImageDraw
        from torch.utils.data import DataLoader
        output_dir = os.path.join(self.output_dir, 'test_patching')
        
        for idx, st in enumerate(self.sts):
            target_patch_size = 224
            with self.subTest(st_object=idx):
                name = ''
                name += st.meta['id']
                st.dump_patches(output_dir, name=name, target_patch_size=target_patch_size)
                
                dataset = H5HESTDataset(os.path.join(output_dir, name + '.h5'), chunk_size=8)
                dataloader = DataLoader(dataset)
                for batch in dataloader:
                    imgs, barcodes, coords = batch['imgs'].squeeze(0), batch['barcodes'], batch['coords'].squeeze(0)
                    for i in range(len(imgs)):
                        img = imgs[i]
                        assert img.shape == (target_patch_size, target_patch_size, 3)
                        barcode = barcodes[i][0]
                        assert barcode.decode('utf-8') in st.adata.obs.index
                        img = Image.fromarray(img.numpy())
                        draw = ImageDraw.Draw(img)
                        text_color = (0, 255, 0)
                        draw.text((0, 0), f'{barcode}, {coords[i]}', fill=text_color)
                        img.save(os.path.join(output_dir, f'{i}_h5_dataset_vis.jpg'))
               
               
    def test_saving(self):
       for idx, st in enumerate(self.sts):
           with self.subTest(st_object=idx):
               name = ''
               name += st.meta['id']
               st.save(os.path.join(self.output_dir, f'test_save_{name}'), save_img=False)

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
    # suite = unittest.TestSuite()
    # suite.addTest(TestHESTData('test_patching'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise Exception('Test failed')