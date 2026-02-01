import os
from dask.distributed import LocalCluster, Client
import pandas as pd

from hest.utils import get_path_from_meta_row, plot_xenium_align_qc, read_parquet_dask, read_parquet_dask_geopandas
from hestcore.wsi import wsi_factory

id_list = [
    'NCBI888',
    'NCBI887',
    'NCBI886',
    'NCBI885',
]

meta_df = pd.read_csv("/home/paul/Downloads/ST H&E datasets - NCBI.csv")
meta_df = meta_df[meta_df['id'].isin(id_list)]

if __name__ == '__main__':
    cluster = LocalCluster(
        "127.0.0.1:8786",
        n_workers=1,
        memory_limit="20GB",
        threads_per_worker=1,
    )
    client = Client(cluster)
    
    
    for _, row in meta_df.iterrows():
        base_path = get_path_from_meta_row(row)
        folder_path = os.path.join(base_path, 'processed')
        print(f"reading {folder_path}")
        
        wsi = wsi_factory(os.path.join(folder_path, f'aligned_fullres_HE.tif'))
        
        df = read_parquet_dask(os.path.join(folder_path, 'aligned_transcripts.parquet'))
        seg_cells = read_parquet_dask_geopandas(os.path.join(folder_path, 'he_cell_seg.parquet'), nb_partitions=30)
        seg_nuc = read_parquet_dask_geopandas(os.path.join(folder_path, 'he_nucleus_seg.parquet'), nb_partitions=30)
        
        # This might be slow for xenium samples
        plot_xenium_align_qc(
            wsi, 
            plot_dir=os.path.join(folder_path, 'qc'), 
            transcript_df=df,
            seg_nuc=seg_nuc,
            seg_cells=seg_cells,
            nb=25, 
            plot_global=True
        )
        
        