from __future__ import annotations

import argparse
import json
import os
from operator import itemgetter
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import yaml
from hestcore.segmentation import get_path_relative
from loguru import logger
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from hest.bench.cpath_model_zoo.inference_models import (
    CustomInferenceEncoder, InferenceEncoder, inf_encoder_factory)

torch.multiprocessing.set_sharing_strategy('file_system')

from hestcore.datasets import H5HESTDataset
from huggingface_hub import snapshot_download

from hest.bench.st_dataset import load_adata
from hest.bench.trainer import train_test_reg
from hest.bench.utils.file_utils import (read_assets_from_h5, save_hdf5,
                                         save_pkl)
from hest.bench.utils.utils import get_current_time, merge_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for linear probing')
### optimizer settings ###
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')
parser.add_argument('--bench_data_root', type=str, help='root directory containing all the datasets')
parser.add_argument('--embed_dataroot', type=str)
parser.add_argument('--weights_root', type=str)
parser.add_argument('--private_weights_root', type=str, default=None)
parser.add_argument('--results_dir', type=str)
parser.add_argument('--exp_code', type=str, default=None)

### specify encoder settings ### 
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')

### specify dataset settings ###
parser.add_argument('--gene_list', type=str, default='var_50genes.json')
parser.add_argument('--method', type=str, default='ridge')
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--kfold', action='store_true', default=False)
parser.add_argument('--benchmark_encoders', action='store_true', default=False)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--dimreduce', type=str, default=None, help='whenever to perform dimensionality reduction before linear probing, can be "PCA" or None')
parser.add_argument('--latent_dim', type=int, default=256, help='dimensionality reduction latent dimension')
parser.add_argument('--encoders', nargs='+', help='All the encoders to benchmark', default=[])
parser.add_argument('--datasets', nargs='+', help='Datasets from bench_data_root to use during benchmark', default=['*'])
parser.add_argument('--config', type=str, help='Path to a benchmark config file, arguments provided in the config file will overwrite the command line args', default=None)
            

def get_path(path):
    src = get_path_relative(__file__, '../../../../')
    if path.startswith('./'):
        new_path = os.path.join(src, path)
    else:
        new_path = path
    return new_path


def benchmark_grid(args, device, model_names, datasets: List[str], save_dir, custom_encoder=None):
    """ Execute predict_folds for each encoders and datasets and dump the results in a nested directory structure """
    
    dataset_perfs = []
    for dataset in datasets:
        bench_data_root = os.path.join(get_path(args.bench_data_root), dataset)
        enc_perfs = []
        for model_name in model_names:
            exp_save_dir = os.path.join(save_dir, dataset, model_name)
            os.makedirs(exp_save_dir, exist_ok=True)
            enc_results = predict_folds(args, exp_save_dir, model_name, dataset, device, bench_data_root)
            
            enc_perfs.append({
                'encoder_name': model_name,
                'pearson_mean': enc_results['pearson_mean'], 
                'pearson_std': enc_results['pearson_std'], 
            })
            
        with open(os.path.join(save_dir, dataset, 'enc_results.json'), 'w') as f:
            enc_perfs = sorted(enc_perfs, key=itemgetter('pearson_mean'), reverse=True)
            json.dump({'results': enc_perfs}, f, sort_keys=True, indent=4)
            

        dataset_perfs.append({
            'dataset_name': dataset,
            'results': enc_perfs
        })
        
    perf_per_enc = {}
    for dataset_perf in dataset_perfs:
        for enc_perf in dataset_perf['results']:
            perf_per_enc[enc_perf['encoder_name']] = perf_per_enc.get(enc_perf['encoder_name'], []) + [enc_perf['pearson_mean']]
          
    row_dicts = []
    for dataset_perf in dataset_perfs:
        row_dict = {}
        row_dict['dataset'] = dataset_perf['dataset_name']
        for result in dataset_perf['results']:
            enc_name = result['encoder_name']
            row_dict[f'{enc_name}_mean'] = result['pearson_mean']
            row_dict[f'{enc_name}_std'] = result['pearson_std']
        row_dicts.append(row_dict)
    df = pd.DataFrame(row_dicts)
    
    df.to_csv(os.path.join(save_dir, 'dataset_results.csv'))
            
    for key, val in perf_per_enc.items():
        perf_per_enc[key] = np.mean(val)
    perf_per_enc = dict(sorted(perf_per_enc.items(), key=lambda item: item[1], reverse=True))
        
    with open(os.path.join(save_dir, 'dataset_results.json'), 'w') as f:
        json.dump({'results': dataset_perfs, 'average': perf_per_enc}, f, sort_keys=True, indent=4)
        

def post_collate_fn(batch):
    """
    Post collate function to clean up batch
    """
    if batch["imgs"].dim() == 5:
        assert batch["imgs"].size(0) == 1
        batch["imgs"] = batch["imgs"].squeeze(0)
    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)
    return batch


def embed_tiles(
    dataloader: DataLoader,
    model: torch.nn.Module,
    embedding_save_path: str,
    device: str,
    precision
):
    """ Extract embeddings from tiles using `encoder` and save to an h5 file """
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch['imgs'].to(device).float().permute((0, 3, 1, 2)) 
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            embeddings = model(imgs)
        if batch_idx == 0:
            mode = 'w'
        else:
            mode = 'a'
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        asset_dict.update({key: np.array(val) for key, val in batch.items() if key != 'imgs'})
        save_hdf5(embedding_save_path,
                  asset_dict=asset_dict,
                  mode=mode)
    return embedding_save_path


def get_bench_weights(weights_root, name):
    local_ckpt_registry = get_path_relative(__file__, 'local_ckpts.json')
    with open(local_ckpt_registry, 'r') as f:
        ckpt_registry = json.load(f)
    if name in ckpt_registry:
        path = ckpt_registry[name]
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(weights_root, path)
    else:
        raise ValueError(f"Please specify the weights path to {name} in {local_ckpt_registry}")

def predict_single_split(train_split, test_split, args, save_dir, dataset_name, model_name, device, bench_data_root):
    if not os.path.isfile(train_split):
        train_split = os.path.join(bench_data_root, 'splits', train_split)
    if not os.path.isfile(test_split):
        test_split = os.path.join(bench_data_root, 'splits', test_split)
    
    train_df = pd.read_csv(train_split)
    test_df = pd.read_csv(test_split)
    
    embedding_dir = os.path.join(get_path(args.embed_dataroot), dataset_name, model_name)
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Embed patches
    logger.info(f"Embedding tiles using {model_name} encoder")
    weights_path = get_bench_weights(args.weights_root, model_name)
    encoder: InferenceEncoder = inf_encoder_factory(model_name)(weights_path)
    precision = encoder.precision
    
    for split in [train_df, test_df]:
        for i in tqdm(range(len(split))):
            sample_id = split.iloc[i]['sample_id']
            tile_h5_path = os.path.join(bench_data_root, split.iloc[i]['patches_path'])
            assert os.path.isfile(tile_h5_path)
            embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
            if not os.path.isfile(embed_path) or args.overwrite:
                
                _ = encoder.eval()
                encoder.to(device)

                tile_dataset = H5HESTDataset(tile_h5_path, chunk_size=args.batch_size, img_transform=encoder.eval_transforms)
                tile_dataloader = torch.utils.data.DataLoader(tile_dataset, 
                                                          batch_size=1, 
                                                          shuffle=False,
                                                          num_workers=args.num_workers)
                
                _ = embed_tiles(tile_dataloader, encoder, embed_path, device, precision)
            else:
                logger.info(f"Skipping {sample_id} as it already exists")


    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    
    # load and gather all data
    all_split_assets = {}
    
    gene_list = args.gene_list

    print(f'using gene_list {gene_list}')
    with open(os.path.join(bench_data_root, gene_list), 'r') as f:
        genes = json.load(f)['genes']
            
    for split_key, split in zip(['train', 'test'], [train_df, test_df]):
        split_assets = {}
        for i in tqdm(range(len(split))):
            sample_id = split.iloc[i]['sample_id']
            embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
            expr_path = os.path.join(bench_data_root, split.iloc[i]['expr_path'])
            assets, _ = read_assets_from_h5(embed_path)
            barcodes = assets['barcodes'].flatten().astype(str).tolist()
            adata = load_adata(expr_path, genes=genes, barcodes=barcodes, normalize=args.normalize)
            assets['adata'] = adata.values
            split_assets = merge_dict(split_assets, assets)
        for key, val in split_assets.items(): 
            split_assets[key] = np.concatenate(val, axis=0)
        
        all_split_assets[split_key] = split_assets  
        logger.info(f"Loaded {split_key} split with {len(split_assets['embeddings'])} samples: {split_assets['embeddings'].shape}")
    
    X_train, y_train = all_split_assets['train']['embeddings'], all_split_assets['train']['adata']
    X_test, y_test = all_split_assets['test']['embeddings'], all_split_assets['test']['adata']
    
    
    if args.dimreduce == 'PCA':
        print('perform PCA dim reduction')
        pipe = Pipeline([('scaler', StandardScaler()), (f'{args.dimreduce}', eval(args.dimreduce)(n_components=args.latent_dim))])
        X_train, X_test = torch.Tensor(pipe.fit_transform(X_train)), torch.Tensor(pipe.transform(X_test))
    
    
    linprobe_results, linprobe_dump = train_test_reg(X_train, X_test, y_train, y_test, random_state=args.seed, genes=genes, method=args.method)
    linprobe_summary = {}
    linprobe_summary.update({'n_train': len(y_train), 'n_test': len(y_test)})
    linprobe_summary.update({key: val for key, val in linprobe_results.items()})
    logger.info(linprobe_summary)
    with open(os.path.join(save_dir, f'results.json'), 'w') as f:
        json.dump(linprobe_results, f, sort_keys=True, indent=4)
    with open(os.path.join(save_dir, f'summary.json'), 'w') as f:
        json.dump(linprobe_summary, f, sort_keys=True, indent=4)
    save_pkl(os.path.join(save_dir, f'inference_dump.pkl'), linprobe_dump)
    return linprobe_results


def merge_fold_results(arr):
    aggr_dict = {}
    for dict in arr:
        for item in dict['pearson_corrs']:
            gene_name = item['name']
            correlation = item['pearson_corr']
            aggr_dict[gene_name] = aggr_dict.get(gene_name, []) + [correlation]
    
    aggr_results = []
    all_corrs = []
    for key, value in aggr_dict.items():
        aggr_results.append({
            "name": key,
            "pearson_corrs": value,
            "mean": np.mean(value),
            "std": np.std(value)
        })
        all_corrs += value
        
    mean_per_split = [d['pearson_mean'] for d in arr]    
        
    return {"pearson_corrs": aggr_results, "pearson_mean": np.mean(mean_per_split), "pearson_std": np.std(mean_per_split), "mean_per_split": mean_per_split}


def benchmark_encoder(encoder: torch.nn.Module, enc_transf, precision, config_path: str) -> dict:
    """ Launch benchmark programmatically """
    
    args = parser.parse_args()

            
    args.config = config_path
    
    
    benchmark(args, encoder=encoder, enc_transf=enc_transf, precision=precision)
        
        
def predict_folds(args, exp_save_dir, model_name, dataset_name, device, bench_data_root):
    split_dir = os.path.join(bench_data_root, 'splits')
    #if not os.path.exists(split_dir):
    #    raise FileNotFoundError(f"{split_dir} doesn't exist, make sure that you specified the ")
    splits = os.listdir(split_dir)
    n_splits = len(splits) // 2
    

    libprobe_results_arr = []
    for i in range(n_splits):
        train_split = os.path.join(split_dir, f'train_{i}.csv')
        test_split = os.path.join(split_dir, f'test_{i}.csv')
        kfold_save_dir = os.path.join(exp_save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        linprobe_results = predict_single_split(train_split, test_split, args, kfold_save_dir, dataset_name, model_name, device=device, bench_data_root=bench_data_root)
        libprobe_results_arr.append(linprobe_results)
        
        
    kfold_results = merge_fold_results(libprobe_results_arr)
    with open(os.path.join(exp_save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)
        
    return kfold_results
            


def benchmark(args, encoder, enc_transf, precision):
    
    if args.config is not None:
        with open(args.config) as stream:
            config = yaml.safe_load(stream)
            
        for key in config:
            if key in args:
                setattr(args, key, config[key])
                
    
    snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=args.weights_root, allow_patterns=['fm_v1/*'])
    
    logger.info(f'Fetch the bench data...')
    snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=args.bench_data_root, ignore_patterns=['fm_v1/*'])
    
    
    logger.info(f'Benchmarking on the following datasets {args.datasets}')
    
    logger.info(f'run parameters {args}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    datasets = args.datasets
    if len(datasets) >= 1 and datasets[0] == '*':
        datasets = os.listdir(get_path(args.bench_data_root))
    
    #### Setup Save Directory ####
    save_dir = get_path(args.results_dir)
    if args.exp_code is None:
        exp_code = f"run_{get_current_time()}"
    else:
        exp_code = args.exp_code + f"::{get_current_time()}"
    save_dir = os.path.join(save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)
    
    encoders = []
    weights_root = args.weights_root
    if encoder is not None:
        custom_encoder = CustomInferenceEncoder(None, 'custom_encoder', encoder, enc_transf, precision) ('custom_encoder', weights_root=weights_root, transforms=enc_transf, model=encoder)
    else:
        custom_encoder = None
        
    encoders += config['encoders']
    
    benchmark_grid(args, device, encoders, datasets, save_dir=save_dir, custom_encoder=custom_encoder)

    
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.config is None:
        parser.error("Please provide --config")
    
    benchmark(args, None, None, None)
    