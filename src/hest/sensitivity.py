import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from typing import Union

def find_common_genes(adata_list):
    """ Find the common genes between multiple AnnData objects

        Args:
            adata_list (list): list of AnnData objects
        
        Returns:
            common_genes (list): list of common genes names
    """
    common_genes = adata_list[0].var_names
    for adata in adata_list[1:]:
        common_genes = np.intersect1d(common_genes, adata.var_names)
    # Remove negative control genes
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
    Filter an AnnData object by the variables listed in gene_list.

    Args:
        adata (sc.AnnData): AnnData object to filter
        gene_list (list): List of variable names.

    Returns:
        sc.AnnData: Filtered AnnData object.
    """
    gene_mask = np.isin(adata.var_names, gene_list)
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
    """ Compute the per-gene sensitivity ratio between two AnnData objects

        Args:
            adata1, adata2 (sc.AnnData): scanpy anndata objects
        
        Returns:
            per_gene_ratio (np.ndarray): sensitivity ratio for each gene
            per_sample_ratio (float): ratio of average sensitivity across all genes
    """

    if adata1.shape[1] != adata2.shape[1]:
        raise ValueError('Number of genes in the two AnnData objects should be the same')
    per_gene_counts1 = compute_median_ignore_zeros(adata1.X, axis=0)
    per_gene_counts2 = compute_median_ignore_zeros(adata2.X, axis=0)

    per_gene_ratio = np.divide(per_gene_counts1, per_gene_counts2, out=np.zeros_like(per_gene_counts1, dtype=np.float64), where=per_gene_counts2!=0)
    per_sample_ratio = np.mean(per_gene_counts1) / np.mean(per_gene_counts2) if np.mean(per_gene_counts2) != 0 else 0

    return per_gene_ratio, per_sample_ratio

def extract_panel_info(adata):
    """ Extract panel information from an AnnData object
        Assumption: All variables targeted by the panel have been detected at least once

        Args:
            adata (sc.AnnData): scanpy anndata object
        
        Returns:
            panel_info (dict): dictionary with the number of genes and controls in the panel
    """
    control_keywords = ['Intergenic', 
                        'UnassignedCodeword', 
                        'NegControlCodeword', 
                        'NegControlProbe', 
                        'DeprecatedCodeword']
    panel_info = {part: sum(part in var_name for var_name in adata.var_names) for part in control_keywords}
    control_variables = sum(panel_info.values())
    target_genes_count = len(adata.var_names) - control_variables
    panel_info['TargetGenes'] = target_genes_count
    return panel_info

def compute_negative_control_codeword_rate(adata):
    """ Compute the negative control codeword rate from an AnnData object
        adjusted by the fraction of codewords that are negative control codewords in the panel

        Args:
            adata (sc.AnnData): scanpy anndata object
        
        Returns:
            nccr (float): negative control codeword rate
    """

    ncc_matrix = adata[:, adata.var_names.str.contains('NegControlCodeword')].X
    ncc_counts = ncc_matrix.sum()
    total_counts = adata.X.sum()
    fraction_ncc_counts = ncc_counts / total_counts if total_counts > 0 else 0

    panel_info = extract_panel_info(adata)
    ncc_var = panel_info.get('NegControlCodeword', 0)
    total_var = len(set(adata.var_names))
    fraction_ncc_var = ncc_var / total_var if total_var > 0 else 0

    nccr = fraction_ncc_counts / fraction_ncc_var if fraction_ncc_var > 0 else 0
    return nccr

def compute_negative_control_probe_rate(adata):
    """ Compute the negative control probe rate from an AnnData object
        adjusted by the fraction of probes that are negative control probes in the panel

        Args:
            adata (sc.AnnData): scanpy anndata object
        
        Returns:
            ncpr (float): negative control probe rate
    """

    ncp_matrix = adata[:, adata.var_names.str.contains('NegControlProbe')].X
    ncp_counts = ncp_matrix.sum()
    total_counts = adata.X.sum()
    fraction_ncp_counts = ncp_counts / total_counts if total_counts > 0 else 0

    panel_info = extract_panel_info(adata)
    ncp_var = panel_info.get('NegControlProbe', 0)
    target_genes_var = panel_info.get('TargetGenes', 0)
    genomics_var = panel_info.get('Intergenic', 0)
    total_var = ncp_var + target_genes_var + genomics_var
    fraction_ncp_var = ncp_var / total_var if total_var > 0 else 0

    ncpr = fraction_ncp_counts / fraction_ncp_var if fraction_ncp_var > 0 else 0
    return ncpr

def extract_panel_info_from_dataframe(transcripts_df):
    """ Extract panel information from a dataframe
        Assumption: All variables targeted by the panel have been detected at least once

        Args:
            transcripts_df (pd.DataFrame): transcript data containing field 'feature_name'
        
        Returns:
            panel_info (dict): dictionary with the number of genes and controls in the panel
    """
    if 'feature_name' not in transcripts_df.columns:
        raise ValueError("The dataframe must contain a 'feature_name' column")
    control_keywords = ['Intergenic', 
                        'UnassignedCodeword', 
                        'NegControlCodeword', 
                        'NegControlProbe', 
                        'DeprecatedCodeword']
    unique_variable_list = transcripts_df['feature_name'].unique()
    panel_info = {part: sum(part in var_name for var_name in unique_variable_list) for part in control_keywords}
    control_variables = sum(panel_info.values())
    target_genes = len(unique_variable_list) - control_variables
    panel_info['TargetGenes'] = target_genes
    return panel_info

def compute_false_positive_rate_metrics(transcripts_df):
    """ Compute the false positive rate metrics from raw transcript data

        Args:
            transcripts_df (pd.DataFrame): transcript data containing field 'feature_name'
        
        Returns:
            nccr (float): negative control codeword rate
            ncpr (float): negative control probe rate
    """
    if 'feature_name' not in transcripts_df.columns:
        raise ValueError("The dataframe must contain a 'feature_name' column")

    panel_info = extract_panel_info_from_dataframe(transcripts_df)

    # Compute NCC rate from raw transcript data
    ncc_counts = transcripts_df[transcripts_df['feature_name'].str.contains('NegControlCodeword')].shape[0]
    total_counts = transcripts_df.shape[0]
    fraction_ncc_counts = ncc_counts / total_counts if total_counts > 0 else 0

    ncc_var = panel_info.get('NegControlCodeword', 0)
    total_var = sum(panel_info.values())
    fraction_ncc_var = ncc_var / total_var if total_var > 0 else 0

    nccr = fraction_ncc_counts / fraction_ncc_var if fraction_ncc_var > 0 else 0

    # Compute NCP rate from raw transcript data
    ncp_counts = transcripts_df[transcripts_df['feature_name'].str.contains('NegControlProbe')].shape[0]
    fraction_ncp_counts = ncp_counts / total_counts if total_counts > 0 else 0
    ncp_var = panel_info.get('NegControlProbe', 0)
    target_genes_var = panel_info.get('TargetGenes', 0)
    genomics_var = panel_info.get('Intergenic', 0)
    total_var = ncp_var + target_genes_var + genomics_var
    fraction_ncp_var = ncp_var / total_var if total_var > 0 else 0

    ncpr = fraction_ncp_counts / fraction_ncp_var if fraction_ncp_var > 0 else 0
    return nccr, ncpr

