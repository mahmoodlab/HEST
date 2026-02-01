
import os

import dask
import pandas as pd
from hest.registration import warp_and_save_xenium_objects
from dask.distributed import LocalCluster, Client, WorkerPlugin


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
    
    root = '/media/paul/ssd2/xenium/Fishing with Two Lines: A Hybrid Approach to Spatial Transcriptomic Discovery/PDLTMA06-11_5K/'

    dapi_transcripts_path = os.path.join(root, 'transcripts.parquet')
    dapi_nucleus_path = os.path.join(root, 'nucleus_boundaries.parquet')
    dapi_cell_path = os.path.join(root, 'cell_boundaries.parquet')
    save_dir = os.path.join(root, 'processed')
    

    warp_and_save_xenium_objects(
        '/home/paul/HEST/results/PDLTMA06-11_5K/2026_01_16_16_34_20/data/_registrar.pickle',
        'morphology_focus_0000.ome.tif',
        save_dir,
        dapi_transcripts=dapi_transcripts_path,
        dapi_cells=dapi_cell_path,
        dapi_nuclei=dapi_nucleus_path,
        use_dask=True,
        verbose=True,
        save_geojson=True
    )
    