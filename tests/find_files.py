import glob
import os
from pathlib import Path
import tifffile
from PIL import Image
from kwimage.im_cv2 import warp_affine, imresize
import numpy as np

# Define the file pattern to search for
file_pattern = '**/**/aligned_fullres_HE.ome.tif'

base_directory = '/mnt/sdb1/paul/data/samples/visium'

paths = []
for root, dirs, files in os.walk(base_directory):
    for file_name in files:
        # Get the full path of the file
        file_path = os.path.join(root, file_name)
        if file_name == 'aligned_fullres_HE.ome.tif':
            paths.append(os.path.join(root,file_name))
        # Process the file as needed (print its full path in this example)
        #print(file_path)
        
for path in paths:
    try:
        img = tifffile.imread(path)
        TARGET_PIXEL_EDGE = 1000
        print('image size is ', img.shape)
        downscale_factor = TARGET_PIXEL_EDGE / np.max(img.shape)
        img = imresize(img, downscale_factor)
        image_pil = Image.fromarray(img)
        sample_name = path.split('visium/')[-1].replace('/', '-')
        print(sample_name)
        image_pil.save(os.path.join('/mnt/sdb1/paul/jpeg', f'{sample_name}.jpeg'))
    except Exception:
        pass