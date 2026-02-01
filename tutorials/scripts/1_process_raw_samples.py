import pandas as pd
from hest.readers import process_raw_samples

from dask.distributed import LocalCluster, Client


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
    cluster = LocalCluster(
        "127.0.0.1:8786",
        n_workers=1,
        memory_limit="30GB",
        threads_per_worker=1,
    )
    client = Client(cluster)
    print('dashboard is available at: ', client.dashboard_link)
    print(client)


    process_raw_samples(
        meta_df,
        save_img=True,
        save_kwargs={
            'save_nuclei_seg': True,
            'save_cell_seg': True,
            'save_transcripts': True
        }
    )
