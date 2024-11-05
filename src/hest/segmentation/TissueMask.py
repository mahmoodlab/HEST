import pickle
from typing import List

import numpy as np
from PIL import Image


class TissueMask:
    
    def __init__(
        self, 
        tissue_mask: np.ndarray, 
        contours_tissue: List, 
        contours_holes: List
    ):
        self.tissue_mask = tissue_mask
        self.contours_tissue = contours_tissue
        self.contours_holes = contours_holes
        

def load_tissue_mask(pkl_path: str, jpg_path: str, width: int, height: int) -> TissueMask:
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
        contours_holes = data['holes']
        contours_tissue = data['tissue']
        with Image.open(jpg_path) as img:
            tissue_mask = np.array(img).copy()
         

        import cv2
        tissue_mask = cv2.resize(tissue_mask, (width, height))
         
        mask = TissueMask(tissue_mask, contours_tissue, contours_holes)
        
    return mask