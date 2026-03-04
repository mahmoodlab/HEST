from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import Dataset
import h5py
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from PIL import Image


class H5PatchDataset(Dataset):
    """Dataset for patch HDF5 files written by HEST/TRIDENT-style patching."""

    def __init__(self, h5_path, img_transform=None):
        self.h5_path = h5_path
        self.img_transform = img_transform
        with h5py.File(self.h5_path, "r") as f:
            self.img_key = "img" if "img" in f else ("imgs" if "imgs" in f else "images")
            self.coords_key = "coords"
            self.barcodes_key = "barcodes" if "barcodes" in f else ("barcode" if "barcode" in f else None)
            self.length = int(f[self.img_key].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            img = f[self.img_key][idx]
            coords = f[self.coords_key][idx]
            if self.barcodes_key is not None:
                barcode = f[self.barcodes_key][idx]
            else:
                barcode = b""

        if isinstance(barcode, np.ndarray) and barcode.shape:
            barcode = barcode[0]
        if isinstance(barcode, bytes):
            barcode = barcode.decode("utf-8")
        else:
            barcode = str(barcode)

        if self.img_transform is not None:
            img_out = self.img_transform(Image.fromarray(img.astype(np.uint8)))
        else:
            img_out = img

        return {
            "imgs": img_out,
            "coords": coords,
            "barcodes": barcode,
        }


def normalize_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    #print(adata.obs)
    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            neighbors_index = adata.obs[((adata.obs['array_row'] >= row - 1) & (adata.obs['array_row'] <= row + 1)) & \
                ((adata.obs['array_col'] >= col - 1) & (adata.obs['array_col'] <= col + 1))].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg
    
    
    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)

    return filtered_adata

def load_adata(expr_path, genes = None, barcodes = None, normalize=False):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]
    if normalize:
        adata = normalize_adata(adata)
    return adata.to_df()
