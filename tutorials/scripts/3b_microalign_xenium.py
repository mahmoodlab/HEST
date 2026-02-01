
import os

import dask
import pandas as pd
from hest.registration import warp_and_save_xenium_objects
from dask.distributed import LocalCluster, Client, WorkerPlugin

from hest.utils import get_path_from_meta_row

id_list = [
    'TENX195',
]

meta_df = pd.read_csv("/home/paul/Downloads/ST H&E datasets - 10XGenomics.csv")
meta_df = meta_df[meta_df['id'].isin(id_list)]

if __name__ == "__main__":
    from valis_hest.registration import init_jvm
    init_jvm(mem_gb=2)
    class JVMPlugin(WorkerPlugin):
        def setup(self, worker):
            from valis_hest.registration import init_jvm
            import jpype
            if not jpype.isJVMStarted():
                init_jvm(mem_gb=1)
            
    
    dask.config.set({
        "distributed.scheduler.worker-ttl": None,
    })

    cluster = LocalCluster(
        "127.0.0.1:8786",
        n_workers=1,
        memory_limit="32GB",
        threads_per_worker=1,
    )
    client = Client(cluster)
    client.register_worker_plugin(JVMPlugin(), name="jvm")
    
    registrar_base = '/home/paul/HEST/results/'
    
    
    for _, row in meta_df.iterrows():
        base_path = get_path_from_meta_row(row)
    
        dapi_transcripts_path = os.path.join(base_path, 'transcripts.parquet')
        dapi_nucleus_path = os.path.join(base_path, 'nucleus_boundaries.parquet')
        dapi_cell_path = os.path.join(base_path, 'cell_boundaries.parquet')
        save_dir = os.path.join(base_path, 'processed')
        
        dirname = list(os.listdir(os.path.join(registrar_base, row['id'])))[0]
        

        warp_and_save_xenium_objects(
            os.path.join(registrar_base, row['id'], dirname, 'data', '_registrar.pickle'),
            'ch0000_dapi.ome.tif',
            #'morphology_focus_0000.ome.tif',
            save_dir,
            dapi_transcripts=dapi_transcripts_path,
            dapi_cells=dapi_cell_path,
            dapi_nuclei=dapi_nucleus_path,
            use_dask=True,
            verbose=True,
            save_geojson=True
        )
        