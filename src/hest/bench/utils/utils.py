

import pdb
import math
import os
from os.path import join as j_
import pickle
import pandas as pd
import datetime
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
import logging


import re

def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"

def extract_patching_info(s):
    match = re.search(r"extracted_mag(\d+)x_patch(\d+)_fp", s)
    mag, patch_size = -1, -1
    if match:
        mag = int(match.group(1))
        patch_size = int(match.group(2))
        return mag, patch_size

def merge_dict(main_dict, new_dict, value_fn = None):
    """
    Merge new_dict into main_dict. If a key exists in both dicts, the values are appended. 
    Else, the key-value pair is added.
    Expects value to be an array or list - if not, it is converted to a list.
    If value_fn is not None, it is applied to each item in each value in new_dict before merging.
    Args:
        main_dict: main dict
        new_dict: new dict
        value_fn: function to apply to each item in each value in new_dict before merging
    """
    if value_fn is None:
        value_fn = lambda x: x
    for key, value in new_dict.items():
        if not isinstance(value, list):
            value = [value]
        value = [value_fn(v) for v in value]
        if key in main_dict:
            main_dict[key] = main_dict[key] + value
        else:
            main_dict[key] = value
    return main_dict

def array2list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def summarize_reulsts(results_dict, ignore_keys = ['folds']):
    summary = {}
    for k, v in results_dict.items():
        if k in ignore_keys: continue
        summary[f"{k}_avg"] = np.mean(v)
        summary[f"{k}_std"] = np.std(v)
    return summary

def seed_torch(seed=7):
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_splits(args, fold_idx=None):
    splits_csvs = {}
    split_names = args.split_names.split(',')
    print(f"Using the following split names: {split_names}")
    for split in split_names:
        if fold_idx is not None:
            split_path = j_(args.split_dir, f'{split}_{fold_idx}.csv')
        else:
            split_path = j_(args.split_dir, f'{split}.csv')
        
        if os.path.isfile(split_path):
            df = pd.read_csv(split_path)#.sample(frac=1, random_state=0).head(25).reset_index(drop=True)
            assert 'Unnamed: 0' not in df.columns
            splits_csvs[split] = df

    return splits_csvs
 

def print_network(net):
    num_params = 0
    num_params_train = 0

    logging.info(str(net))
    # print(str(net))
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    logging.info(f'Total number of parameters: {num_params}')
    logging.info(f'Total number of trainable parameters: {num_params_train}')
