import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import issparse
from typing import Union

def find_common_genes(adata_list):
    """ Find the common target genes across multiple AnnData objects, excluding controls.
        ASSUMPTION: Controls are labeled with specific keywords.

        Args:
            adata_list (list): list of AnnData objects
        
        Returns:
            common_genes (list): list of common gene names present in all AnnData objects
    """
    common_genes = adata_list[0].var_names
    for adata in adata_list[1:]:
        common_genes = np.intersect1d(common_genes, adata.var_names)
    # Remove negative controls
    control_keywords = ['NegControlCodeword', 
                        'NegControlProbe', 
                        'Intergenic', 
                        'UnassignedCodeword', 
                        'DeprecatedCodeword']
    common_genes = [gene for gene in common_genes 
                    if not any(keyword in gene for keyword in control_keywords)]
    return common_genes

def filter_adata_by_gene_names(adata: sc.AnnData, gene_list: list) -> sc.AnnData:
    """
    Filter an AnnData object to retain only the genes specified in the provided gene list.

    Args:
        adata (sc.AnnData): AnnData object to filter
        gene_list (list): list of gene names (or variable names) to retain

    Returns:
        sc.AnnData: filtered AnnData object containing only the genes specified. 
    """
    gene_mask = np.isin(adata.var_names, gene_list)
    if not np.any(gene_mask):
        raise ValueError("None of the genes in the gene_list are present in the AnnData object.")
    adata_filtered = adata[:, gene_mask]
    return adata_filtered

def compute_median_ignore_zeros(data: Union[np.ndarray, "scipy.sparse.spmatrix"], axis: int = 0) -> np.ndarray:
    """ 
    Compute the median of non-zero elements in a matrix along a given axis.

    Args:
        data (Union[np.ndarray, scipy.sparse.spmatrix]): Input matrix (dense or sparse)
        axis (int): Axis along which to compute the median. Must be 0 or 1.

    Returns:
        np.ndarray: Median of non-zero elements along the given axis.
    """
    if issparse(data):
        data = np.nan_to_num(data.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    def median_non_zero(arr: np.ndarray) -> float:
        non_zero_elements = arr[arr != 0]
        return np.median(non_zero_elements) if non_zero_elements.size > 0 else 0

    return np.apply_along_axis(median_non_zero, axis, data)

def paired_sensitivity(adata1, adata2):
    """ Compute the per-gene relative sensitivity between two samples. 
        Computes the ratio of detection sensitivity between two AnnData objects 
        for each gene and the overall ratio of the mean sensitivities across all genes.

        Args:
            adata1, adata2 (sc.AnnData): scanpy anndata objects
        
        Returns:
            per_gene_ratio (np.ndarray): sensitivity ratio for each gene between the two samples
            per_sample_ratio (float): ratio of the average sensitivity across all genes between the two samples
    """

    if adata1.shape[1] != adata2.shape[1]:
        raise ValueError('Number of genes in the two AnnData objects should be the same')
    per_gene_counts1 = compute_median_ignore_zeros(adata1.X, axis=0)
    per_gene_counts2 = compute_median_ignore_zeros(adata2.X, axis=0)

    per_gene_ratio = np.divide(per_gene_counts1, per_gene_counts2, out=np.zeros_like(per_gene_counts1, dtype=np.float64), where=per_gene_counts2!=0)
    per_sample_ratio = np.mean(per_gene_counts1) / np.mean(per_gene_counts2) if np.mean(per_gene_counts2) != 0 else 0

    return per_gene_ratio, per_sample_ratio