import os
import unittest
import warnings
from datetime import datetime
from os.path import join as _j
import zipfile

from hest.path_utils import get_path_relative
from hest.trident_compat import CucimWarningSingleton
from huggingface_hub import snapshot_download
from tqdm import tqdm

MAX_HEST_IMPORT_S = 2
start_time = datetime.now()
import hest
end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
if elapsed_time > MAX_HEST_IMPORT_S:
    raise ImportError(f"Importing 'hest' took too long ({elapsed_time:.2f} seconds). Maximum allowed time is {MAX_HEST_IMPORT_S} seconds. Please, keep large large imports conditional")

from hest.autoalign import autoalign_visium
from hest.readers import VisiumReader, pool_bins_visiumhd
from hest.HESTData import ensembl_id_to_gene
from hest.utils import load_image

def download_hest(patterns, local_dir):
    repo_id = 'MahmoodLab/hest'
    snapshot_download(repo_id=repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_dir)

    seg_dir = os.path.join(local_dir, 'cellvit_seg')
    if os.path.exists(seg_dir):
        print('Unzipping cell vit segmentation...')
        for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
            path_zip = os.path.join(seg_dir, filename)
                        
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(seg_dir)


class TestVisiumHDPooling(unittest.TestCase):
    def test_count_conservation(self):
        from itertools import product
        from tempfile import TemporaryDirectory

        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy.sparse import csr_matrix, csc_matrix, issparse

        # Each destination has repeated indices and a final zero-count source bin.
        counts = np.array([
            [1, 10, 0], [2, 20, 0], [3, 30, 0], [0, 0, 0],
            [4, 40, 0], [5, 50, 0], [6, 60, 0], [0, 0, 0],
        ], dtype=np.float32)
        obs = pd.DataFrame({
            'pxl_col_in_fullres': [8, 24, 8, 24, 136, 152, 136, 152],
            'pxl_row_in_fullres': [8, 8, 24, 24, 8, 8, 24, 24],
        }, index=[str(i) for i in range(8)])
        genes = ['gene_b', 'gene_a', 'zero_gene']
        orders = [np.arange(8), np.arange(8)[::-1], np.array([0, 4, 1, 5, 2, 6, 3, 7])]

        with TemporaryDirectory() as tmp:
            for storage, backed, chunk_len, order in product(
                (np.array, csr_matrix, csc_matrix), (False, True), (1, 2, 3, 50000), orders
            ):
                with self.subTest(storage=storage.__name__, backed=backed,
                                  chunk_len=chunk_len, order=order.tolist()):
                    adata = ad.AnnData(
                        storage(counts[order]), obs=obs.iloc[order].copy(),
                        var=pd.DataFrame(index=genes),
                    )
                    if backed:
                        path = os.path.join(tmp, 'counts.h5ad')
                        adata.write_h5ad(path)
                        adata = ad.read_h5ad(path, backed='r')
                    try:
                        pooled = pool_bins_visiumhd(adata, pixel_size=1.0, chunk_len=chunk_len)
                        np.testing.assert_array_equal(pooled.X, [[6, 60, 0], [15, 150, 0]])
                        self.assertEqual(pooled.X.sum(), counts.sum())
                        self.assertEqual(pooled.var_names.tolist(), genes)
                        np.testing.assert_array_equal(pooled.obsm['spatial'], [[72, 72], [200, 72]])
                        np.testing.assert_array_equal(pooled.obs['array_col'], [0, 1])
                        np.testing.assert_array_equal(pooled.obs['array_row'], [0, 0])
                        remaining = adata.X[:]
                        if issparse(remaining):
                            remaining = remaining.toarray()
                        np.testing.assert_array_equal(remaining, counts[order])
                    finally:
                        if backed:
                            adata.file.close()


class TestHESTReader(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.cur_dir = get_path_relative(__file__, '')
        cur_dir = self.cur_dir
        required_paths = [
            _j(cur_dir, './assets/WSA_LngSP9258463.jpg'),
            _j(cur_dir, './assets/filtered_feature_bc_matrix.h5'),
            _j(cur_dir, './assets/spatial'),
        ]
        missing_paths = [p for p in required_paths if not os.path.exists(p)]
        if missing_paths:
            raise unittest.SkipTest(f"Missing local reader test assets: {missing_paths}")
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
        
        token = (
            os.getenv('HF_TOKEN')
            or os.getenv('HUGGINGFACE_HUB_TOKEN')
            or os.getenv('HF_READ_TOKEN_PAUL')
        )
        if token is not None:
            login(token=token)
        download = True
        
        id_list = ['TENX24', 'SPA154']
        
        if download:
            local_dir = os.path.join(cur_dir, 'hest_data_test')
            
            ids_to_query = id_list
            list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
            download_hest(list_patterns, local_dir)
            
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
                #st.segment_tissue(method='deep', verbose=True, fast_mode=True)
                #st.save_tissue_contours(self.output_dir, name=f'deep_{idx}')
                #st.save_tissue_vis(self.output_dir, name=f'deep_{idx}')
                
                st.segment_tissue(method='otsu')
                st.save_tissue_contours(self.output_dir, name=f'otsu_{idx}')
                st.save_tissue_vis(self.output_dir, name=f'otsu_{idx}')


    def test_spatialdata(self):
        for idx, st in enumerate(self.sts):
            with self.subTest(st_object=idx):
                name = ''
                name += st.meta['id']
                spd = st.to_spatial_data()
                print(spd)


    def test_patching(self):
        """ Save patches as .h5 then load with H5PatchDataset """
        from hest.bench.st_dataset import H5PatchDataset
        from PIL import Image, ImageDraw
        from torch.utils.data import DataLoader
        output_dir = os.path.join(self.output_dir, 'test_patching')
        
        for idx, st in enumerate(self.sts):
            target_patch_size = 224
            with self.subTest(st_object=idx):
                name = ''
                name += st.meta['id']
                st.dump_patches(output_dir, name=name, target_patch_size=target_patch_size)
                
                dataset = H5PatchDataset(os.path.join(output_dir, name + '.h5'))
                dataloader = DataLoader(dataset, batch_size=8)
                for batch in dataloader:
                    imgs, barcodes, coords = batch['imgs'], batch['barcodes'], batch['coords']
                    for i in range(len(imgs)):
                        img = imgs[i]
                        assert img.shape == (target_patch_size, target_patch_size, 3)
                        barcode = barcodes[i]
                        assert barcode in st.adata.obs.index
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
    suite.addTests(loader.loadTestsFromTestCase(TestVisiumHDPooling))
    # suite = unittest.TestSuite()
    #suite.addTest(TestHESTData('test_spatialdata'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise Exception('Test failed')
