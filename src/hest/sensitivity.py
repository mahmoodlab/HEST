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

def get_var_names(data):
    """ Extract the variable names (e.g., gene names, controls) from a sample.

        Args:
        data (sc.AnnData or pd.DataFrame): can be one of the following:
            - AnnData object containing gene expression matrix
            - DataFrame containing raw transcript data
        
        Returns:
        var_names (list): list of gene names and controls codewords
    """
    if isinstance(data, sc.AnnData):
        return data.var_names.tolist()
    elif isinstance(data, pd.DataFrame):
        if 'feature_name' not in data.columns:
            raise ValueError("The dataframe must contain a 'feature_name' column")
        return data['feature_name'].unique().tolist()
    else:
        raise TypeError("Input must be either an AnnData object or a DataFrame")

def get_num_genes(data):
    """ Calculate the number of target genes from the data by excluding controls.
        ASSUMPTION: all variables targeted by the panel have been detected at least once
        and controls are labeled with specific keywords.

        Args:
        data (sc.AnnData or pd.DataFrame): can be one of the following:
            - AnnData object containing gene expression matrix
            - DataFrame containing raw transcript data
        
        Returns:
        num_genes (int): number of target genes
    """
    var_names = get_var_names(data)
    control_keywords = ['Intergenic', 
                        'UnassignedCodeword', 
                        'NegControlCodeword', 
                        'NegControlProbe', 
                        'DeprecatedCodeword']
    num_genes = len([var_name for var_name in var_names if not any(keyword in var_name for keyword in control_keywords)])
    return num_genes

def get_num_controls(data):
    """ Calculate the number of control codewords from the data.
        Counts the occurrences of each type of control variable based on predefined keywords.
        ASSUMPTION: all variables targeted by the panel have been detected at least once
        and controls are labeled with specific keywords.

        Args:
        data (sc.AnnData or pd.DataFrame): can be one of the following:
            - AnnData object containing gene expression matrix
            - DataFrame containing raw transcript data
        
        Returns:
        num_controls (dict): number of each type of control codeword
    """
    var_names = get_var_names(data)
    control_keywords = ['Intergenic', 
                        'UnassignedCodeword', 
                        'NegControlCodeword', 
                        'NegControlProbe', 
                        'DeprecatedCodeword']
    num_controls = {part: sum(part in var_name for var_name in var_names) for part in control_keywords}
    return num_controls

def compute_negative_control_codeword_rate(data):
    """ Compute the Negative Control Codeword Rate (NCCR) for a given sample. 
        NCCR is the false positive rate of the decoding algorithm adjusted to the proportion
        of negative control codewords in the panel.

        Args:
        data (sc.AnnData or pd.DataFrame): can be one of the following:
            - AnnData object containing gene expression matrix
            - DataFrame containing raw transcript data
        
        Returns:
        nccr (float): Negative Control Codeword Rate
    """
    if isinstance(data, sc.AnnData):
        ncc_counts = data[:, data.var_names.str.contains('NegControlCodeword')].X.sum()
        total_counts = data.X.sum()
        
    elif isinstance(data, pd.DataFrame):
        if 'feature_name' not in data.columns:
            raise ValueError("Dataframe must contain a 'feature_name' column")
        ncc_counts = data[data['feature_name'].str.contains('NegControlCodeword')].shape[0]
        total_counts = data.shape[0]
    else:
        raise TypeError("Input must be either an AnnData object or a DataFrame")
    
    fraction_ncc_counts = ncc_counts / total_counts if total_counts > 0 else 0
    
    num_genes = get_num_genes(data)
    num_controls = get_num_controls(data)

    ncc_var = num_controls.get('NegControlCodeword', 0)
    total_var = sum(num_controls.values()) + num_genes
    fraction_ncc_var = ncc_var / total_var if total_var > 0 else 0

    nccr = fraction_ncc_counts / fraction_ncc_var if fraction_ncc_var > 0 else 0
    return nccr

def compute_negative_control_probe_rate(data):
    """ Compute the Negative Control Probe Rate (NCCR) for a given sample. 
        NCPR is the false positive rate of the transcript signal adjusted to the proportion
        of probes that are negative control probes in the panel.

        Args:
        data (sc.AnnData or pd.DataFrame): can be one of the following:
            - AnnData object containing gene expression matrix
            - DataFrame containing raw transcript data
        
        Returns:
        ncpr (float): Negative Control Probe Rate
    """
    if isinstance(data, sc.AnnData):
        ncp_counts = data[:, data.var_names.str.contains('NegControlProbe')].X.sum()
        total_counts = data.X.sum()
    elif isinstance(data, pd.DataFrame):
        if 'feature_name' not in data.columns:
            raise ValueError("Dataframe must contain a 'feature_name' column")
        ncp_counts = data[data['feature_name'].str.contains('NegControlProbe')].shape[0]
        total_counts = data.shape[0]
    else:
        raise TypeError("Input must be either an AnnData object or a DataFrame")
    
    fraction_ncp_counts = ncp_counts / total_counts if total_counts > 0 else 0

    num_genes = get_num_genes(data)
    num_controls = get_num_controls(data)

    ncp_var = num_controls.get('NegControlProbe', 0)
    genomics_var = num_controls.get('Intergenic', 0)
    total_var = ncp_var + num_genes + genomics_var
    fraction_ncp_var = ncp_var / total_var if total_var > 0 else 0

    ncpr = fraction_ncp_counts / fraction_ncp_var if fraction_ncp_var > 0 else 0
    return ncpr