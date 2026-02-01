
import os

import pandas as pd

from hest.registration import register_dapi_he
from hest.utils import get_path_from_meta_row

id_list = [
    'TENX202',
    'TENX201',
    'TENX200',
    'TENX199',
    'TENX198',
    'TENX197',
]

meta_df = pd.read_csv("/home/paul/Downloads/ST H&E datasets - 10XGenomics.csv")
meta_df = meta_df[meta_df['id'].isin(id_list)]

if __name__ == "__main__":
    for _, row in meta_df.iterrows():
        base_path = get_path_from_meta_row(row)
        
        dapi_path = None
        if os.path.exists(os.path.join(base_path, 'morphology_focus', 'ch0000_dapi.ome.tif')):
            dapi_path = os.path.join(base_path, 'morphology_focus', 'ch0000_dapi.ome.tif')
        elif os.path.exists(os.path.join(base_path, 'morphology_focus', 'morphology_focus_0000.ome.tif')):
            dapi_path = os.path.join(base_path, 'morphology_focus', 'morphology_focus_0000.ome.tif')
        else:
            raise ValueError("No valid dapi path found,")
    
        register_dapi_he(
            os.path.join(base_path, 'processed/aligned_fullres_HE.tif'),
            dapi_path, 
            os.path.join("results", row['id']),
            name = None,
            max_non_rigid_registration_dim_px=10000,
            micro_rigid_registrar_cls=None,
            micro_rigid_registrar_params={},
            micro_reg=True,
            check_for_reflections=False
        )