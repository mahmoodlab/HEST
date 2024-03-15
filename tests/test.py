import scanpy as sc
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile
import numpy as np
from kwimage.im_cv2 import imresize

from src.hiSTloader import read_10x_visium, extract_image_patches, visualize_patches, filter_st_data, extract_patch_expression_pairs, read_10x_xenium


#adata, df, img = read_10x_visium('/mnt/ssd/paul/data/visium/10x/Adult Mouse Olfactory Bulb/Visium_Mouse_Olfactory_Bulb_filtered_feature_bc_matrix.h5',
#    '/mnt/ssd/paul/data/visium/10x/Adult Mouse Olfactory Bulb/spatial/tissue_positions.csv',
#    '/mnt/ssd/paul/data/visium/10x/Adult Mouse Olfactory Bulb/Visium_Mouse_Olfactory_Bulb_image.tif',
#    '/mnt/ssd/paul/data/visium/10x/Adult Mouse Olfactory Bulb/Visium_Mouse_Olfactory_Bulb_alignment_file.json'
#)

#sc.pp.calculate_qc_metrics(adata, inplace=True)
#plt.rcParams["figure.figsize"] = (8, 8)
#sc.pl.spatial(adata, img_key="fullres", color=["total_counts"])


img = tifffile.imread('/mnt/ssd/paul/data/Visium_Fresh_Frozen_Adult_Mouse_Brain_image.tif')
#img = tifffile.imread('/mnt/ssd/paul/data/Visium_Mouse_Olfactory_Bulb_image.tif')
#img = imresize(img, 0.1)
img = imresize(img, dsize=(1333, 1333))
print(img.shape)
cimg = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
#cimg = cv.array

circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1,10,
                            param1=300,param2=10,minRadius=0,maxRadius=20)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
 
cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()


#anndata, df, img = read_10x_visium('/mnt/ssd/paul/data/visium/10x/Adult Mouse Brain Coronal Section (Fresh Frozen)/V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5',
#    '/mnt/ssd/paul/data/visium/10x/Adult Mouse Brain Coronal Section (Fresh Frozen)/spatial/tissue_positions.csv',
#    '/mnt/ssd/paul/data/visium/10x/Adult Mouse Brain Coronal Section (Fresh Frozen)/V1_Adult_Mouse_Brain_image.tif'
#)

#print(anndata.uns)



#anndata = sc.read_visium(
#    '/mnt/ssd/paul/data/visium/10x/Adult Mouse Brain Coronal Section (Fresh Frozen)', 
#   count_file='V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5', load_images=True, source_image_path='tissue_hires_image.png')

#print(df)
#print(anndata.obsm['spatial'])
#print(anndata.var)"""



"""
patches = extract_image_patches(df, img, patch_size = 200)

print('extract pairs')
pairs_df = extract_patch_expression_pairs(anndata, df, img)
print(pairs_df)

visualize_patches(patches[:5])
"""


#adata, img = read_10x_xenium(
#    '/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/cell_feature_matrix.h5',
#    '/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/cells.csv',
#    '/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/Xenium_V1_hSkin_nondiseased_section_1_FFPE_he_image.ome.tif',
#    '/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/Xenium_V1_hSkin_nondiseased_section_1_FFPE_he_imagealignment.csv'
#)

#adata, img = read_10x_xenium(
#    feature_matrix_path='/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/sample2/cell_feature_matrix.h5',
#    cell_csv_path='/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/sample2/cells.csv',
#    img_path='/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/sample2/Xenium_V1_hSkin_nondiseased_section_2_FFPE_he_image.ome.tif',
#    alignment_file_path='/mnt/ssd/paul/data/xenium/10x/Human Skin Data with Xenium Human Multi-Tissue and Cancer Panel/sample2/Xenium_V1_hSkin_nondiseased_section_2_FFPE_he_imagealignment.csv'
#)

#adata, img = read_10x_xenium(
#    feature_matrix_path='/mnt/ssd/paul/data/xenium/10x/Human Liver Data with Xenium Human Multi-Tissue and Cancer Panel/nondiseased/cell_feature_matrix.h5',
#    cell_csv_path='/mnt/ssd/paul/data/xenium/10x/Human Liver Data with Xenium Human Multi-Tissue and Cancer Panel/nondiseased/cells.csv',
#    img_path='/mnt/ssd/paul/data/xenium/10x/Human Liver Data with Xenium Human Multi-Tissue and Cancer Panel/nondiseased/Xenium_V1_hLiver_nondiseased_section_FFPE_he_image.ome.tif',
#    alignment_file_path='/mnt/ssd/paul/data/xenium/10x/Human Liver Data with Xenium Human Multi-Tissue and Cancer Panel/nondiseased/Xenium_V1_hLiver_nondiseased_section_FFPE_he_imagealignment.csv'
#)

#print(adata)

#sc.pp.calculate_qc_metrics(adata, inplace=True)
#print(adata.var_names)


#plt.rcParams["figure.figsize"] = (8, 8)
#sc.pl.spatial(adata, img_key="fullres", color=["ACE2"])


while True:
     pass