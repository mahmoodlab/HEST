import numpy as np
import itertools
from tqdm import tqdm
from sklearn.metrics import (balanced_accuracy_score, 
                             cohen_kappa_score, 
                             classification_report, 
                             accuracy_score, 
                             f1_score,
                             roc_auc_score)
from st_bench.utils.utils import merge_dict

def swap_at_ids(arr1, arr2, ids):
    """
    Swap rows of arr1 and arr2 at indices ids
    """
    temp = arr1[ids]
    arr1[ids] = arr2[ids]
    arr2[ids] = temp
    return arr1, arr2

def permutation_test(scores_1, 
                     scores_2, 
                     n = 1000, 
                     seed = 1, 
                     metric_fn = lambda x: {'mean': x.mean(axis=0)},
                     alt = 'two-sided'):
    """
    Permutation test for difference in scores_1 and scores_2
    args:
        scores_1: numpy array 
        scores_2: numpy array
        n: number of permutations to sample
        seed: random seed
        metric_fn: function to compute metric of interest
        alt: alternative hypothesis
    """
    n_obs = len(scores_1)

    # observed diff 
    metrics_1 = metric_fn(scores_1)
    metrics_2 = metric_fn(scores_2)
    metric_keys = metrics_1.keys()
    obs_diff_dict = {key: metrics_1[key] - metrics_2[key] for key in metric_keys}
    
    np.random.seed(seed)
    sampled_diff_dict = {}
    
    if n < 0: # n == -1 ----> generate all possible permutations (usually too expensive)
        for ids in tqdm(iter(itertools.product([0, 1], repeat=n_obs))):
            swap_ids = np.where(np.array(ids) == 1)[0]
            temp1, temp2 = swap_at_ids(scores_1.copy(), scores_2.copy(), swap_ids)
            metrics_1 = metric_fn(temp1)
            metrics_2 = metric_fn(temp2)
            sampled_diff_dict = merge_dict(sampled_diff_dict, {key: metrics_1[key] - metrics_2[key] for key in metric_keys})
    else: # else, randomly sample n permutations
        for i in tqdm(range(n)):
            # generate random binary array: 0 means don't swap, 1 means swap
            ids = np.random.randint(2, size=n_obs)
            swap_ids = np.where(ids == 1)[0]
            temp1, temp2 = swap_at_ids(scores_1.copy(), scores_2.copy(), swap_ids)
            metrics_1 = metric_fn(temp1)
            metrics_2 = metric_fn(temp2)
            sampled_diff_dict = merge_dict(sampled_diff_dict, {key: metrics_1[key] - metrics_2[key] for key in metric_keys})
            
    # # sampled_diffs is length n_obs
    # sampled_diffs = np.stack(sampled_diffs)
    return_dict = {}
    for key in metric_keys:
        obs_diff = obs_diff_dict[key]
        sampled_diffs = np.array(sampled_diff_dict[key])
        if alt == 'two-sided':
            sampled_diffs = np.abs(sampled_diffs)
            obs_diff = np.abs(obs_diff)
            # p-value is proportion of shuffled difference values that are greater in magnitude than the observed 
            p_value = (obs_diff <= sampled_diffs).sum(axis=0) / sampled_diffs.shape[0]
        elif alt == '>':
            # p-value is proportion of shuffled difference values that are greater than the observed 
            p_value = (obs_diff <= sampled_diffs).sum(axis=0) / sampled_diffs.shape[0]
        elif alt == '<':
            # p-value is proportion of shuffled difference values that are smaller than the observed 
            p_value = (sampled_diffs <= obs_diff).sum(axis=0) / sampled_diffs.shape[0]
        return_dict[key] = {'obs_diff': obs_diff, 
                            'sampled_diffs': sampled_diffs, 
                            'p_value': p_value}
    return return_dict

def sweep_classification_metrics(all_probs, all_labels, all_preds=None, n_classes=None, return_report=True):
    if n_classes is None:
        n_classes = all_probs.shape[1]

    if all_preds is None:
        all_preds = all_probs.argmax(axis=1)
        
    if n_classes == 2:
        all_probs = all_probs[:,1]
        roc_kwargs = {}
    else:
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}        
    
    bacc = balanced_accuracy_score(all_labels, all_preds)
    weighted_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    kappa = cohen_kappa_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, **roc_kwargs)
    except ValueError:
        roc_auc = np.nan

    results = {'acc': acc, 
               'bacc': bacc, 
               'kappa': kappa,
               'weighted_kappa': weighted_kappa, # 'quadratic
               'roc_auc': roc_auc,
               'weighted_f1': weighted_f1}
    if return_report:
        cls_rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0) 
        results['cls_rep'] = cls_rep
    return results

def bootstrap_CI(preds=None, targets=None, metric_fn=None, n=1000, alpha=0.95, seed=1, clamp_dict={}):
    """
    Compute confidence interval for a given metric using bootstrap resampling.
    Args:
        preds: predictions
        targets: ground-truth labels
        nonreduced_scores: scores that are already computed for each sample, but not reduced to a single value
        metric_fn: metric function to compute the score given preds and targets
        n: number of bootstrap samples
        alpha: confidence level
        seed: random seed
    """
    np.random.seed(seed)

    # perform bootstrap
    all_scores = {}
    for i in tqdm(range(n)):
        sample_ids = np.random.choice(len(preds), 
                                        len(preds), 
                                        replace=True)
        sample_preds = preds[sample_ids]
        if targets is not None:
            sample_targets = targets[sample_ids]
        else:
            sample_targets = None # metric_fn may not need targets
        sample_scores = metric_fn(sample_preds, sample_targets)
        assert isinstance(sample_scores, dict), "metric_fn should return a dict"
        merge_dict(all_scores, sample_scores)

    remove_keys = []
    for key, val in all_scores.items():
        if isinstance(val[0], dict):
            remove_keys.append(key)
            print(f"Warning: metric {key} consists of dicts. Removed.")
    for key in remove_keys:
        del all_scores[key]

    # compute confidence interval
    return_dict = {}
    for k in all_scores.keys():
        lower_clamp, upper_clamp = clamp_dict.get(k, (0.0, 1.0))
        scores = np.array(all_scores[k])
        p = ((1.0-alpha)/2.0) * 100
        lower = max(lower_clamp, np.nanpercentile(scores, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(upper_clamp, np.nanpercentile(scores, p))
        return_dict[k] = (lower, upper)
    return return_dict  