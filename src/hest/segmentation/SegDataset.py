import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from hest.wsi import WSIPatcher

class SegFileDataset(Dataset):
    masks = []
    patches = []
    coords = []
    
    def __init__(self, root_path, transform):
        self._load_paths(root_path)
        
        self.transform = transform
        
    def _load_paths(self, root_path):
        self.mask_paths = []
        self.patch_paths = []
        self.coords = []
        for mask_filename in tqdm(os.listdir(root_path)):
            name = mask_filename.split('.')[0]
            pxl_x, pxl_y = int(name.split('_')[0]), int(name.split('_')[1])
            self.patch_paths.append(os.path.join(root_path, mask_filename))
            self.coords.append([pxl_x, pxl_y])
                        

    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, index):
            
        with Image.open(self.patch_paths[index]) as patch:
            patch = np.array(patch)
            coord = self.coords[index]
            
        sample = patch
        
        if self.transform:
            sample = self.transform(sample)

        return sample, coord
    
    
class SegWSIDataset(Dataset):
    
    def __init__(self, patcher: WSIPatcher, transform):
        self.patcher = patcher
        
        self.transform = transform
                              

    def __len__(self):
        return len(self.patcher)
    
    def __getitem__(self, index):
        tile, x, y = self.patcher[index]
        
        if self.transform:
            tile = self.transform(tile)

        return tile, (x, y)