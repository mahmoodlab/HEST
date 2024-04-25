import pandas as pd
from src.hest.old_st import *
from src.hest.helpers import extract_patch_expression_pairs, visualize_patches, extract_image_patches, download_from_meta_df
from valis import registration
import tifffile
from kwimage.im_cv2 import imresize

def main():
    
    
    #img = tifffile.imread('/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Visium CytAssist Gene Expression Libraries of Post-Xenium Mouse Brain (FF)/Control, Replicate 1/CytAssist_FreshFrozen_Mouse_Brain_Rep1_tissue_image.btf')
    #img2 = imresize(img, 0.025)
    #tifffile.imwrite('/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Visium CytAssist Gene Expression Libraries of Post-Xenium Mouse Brain (FF)/Control, Replicate 1/downscaled.tif', img2)
    

    slide_src_dir = "/mnt/ssd/paul/ST-histology-loader/data/samples/visium/Visium CytAssist Gene Expression Libraries of Post-Xenium Mouse Brain (FF)/Control, Replicate 1/test"
    results_dst_dir = "./slide_registration_example"
    registered_slide_dst_dir = "./slide_registration_example/registered_slides"
    reference_slide = "CytAssist_FreshFrozen_Mouse_Brain_Rep1_image.tif"

    # Create a Valis object and use it to register the slides in slide_src_dir, aligning *towards* the reference slide.
    registrar = registration.Valis(slide_src_dir, results_dst_dir, reference_img_f=reference_slide)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Perform micro-registration on higher resolution images, aligning *directly to* the reference image
    registrar.register_micro(max_non_rigid_registration_dim_px=2000, align_to_reference=True)
    
    
if __name__ == "__main__":
    main()
    