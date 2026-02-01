import pandas as pd

from hest.readers import process_meta_df_cellvit


id_list = [
    'TENX202',
    'TENX201',
    'TENX200',
    'TENX199',
    'TENX198',
    'TENX197',
]


if __name__ == '__main__':
    df = pd.read_csv("/home/paul/Downloads/ST H&E datasets - 10XGenomics.csv")


    df = df[df['id'].isin(id_list)]

    process_meta_df_cellvit(df, {'gpu_ids': [0], 'batch_size': 1, 'model': 'CellViT-256-x40.pth'})
