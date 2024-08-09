from __future__ import annotations

import os
import pickle
from abc import abstractmethod

import numpy as np
import pandas as pd
from loguru import logger

from hest.HESTData import unify_gene_names


def reduce_dim(X, indices=None, harmony=False):
    import umap
    from sklearn.decomposition import PCA
    
    print('perform PCA...')
    pca = PCA(n_components=50)
    comps = pca.fit_transform(X)
    
    if harmony:
        print('perform harmony...')
        meta_df = pd.DataFrame(indices, columns=['dataset'])
        meta_df['dataset'] = meta_df['dataset'].astype(str)
        
        import harmonypy as hm
        vars_use = ['dataset']
        
        X = hm.run_harmony(comps, meta_df, vars_use, verbose=False).Z_corr.transpose()
    else:
        X = comps
    
    print('perform UMAP...')
    reducer = umap.UMAP(verbose=False)
    embedding = reducer.fit_transform(X)
    return embedding


def get_per_cluster_types(cells: sc.AnnData, cluster_key='Cluster', type_key='cell_type_pred', pct=False):
    clusters = np.unique(cells.obs[cluster_key])
    ls = []
    for cluster in clusters:
        cell_types = cells[cells.obs[cluster_key] == cluster].obs[type_key]
        types, counts = np.unique(cell_types, return_counts=True)
        freqs = list(zip(list(types), list(counts)))
        freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
        if pct:
            s = np.array([a[1] for a in freqs]).sum()
            freqs = [(a[0], str(round(100 * a[1] / s)) + '%') for a in freqs if (100 * a[1] / s) > 1]
        ls.append([cluster, freqs])
    return ls


def plot(plot_name, embedding, cell_types, indices=None):
    import seaborn as sns
    from matplotlib import pyplot as plt

    print('find uniques...')
    
    cell_types = np.array([str(s) for s in cell_types])
    names, inverse = np.unique(cell_types, return_inverse=True)
    plt.figure(figsize=(10, 6))


    if indices is None:
        indices = np.array([0 for _ in range(len(cell_types))])
    
    palettes = ['tab10', 'tab10']
    markers = ['o', '^']
    for i in np.unique(indices):
        obs_names = cell_types
        
        sub_emb = embedding[indices==i, :]
        sub_obs_names = obs_names[indices==i]
        n = len(sub_emb)
        k = 1000
        idx = np.random.choice(np.arange(n), size=k, replace=False)
        sub_emb = sub_emb[idx, :]
        sub_obs_names = sub_obs_names[idx]
        
        sns.scatterplot(
            x=sub_emb[:, 0],
            y=sub_emb[:, 1],
            hue=sub_obs_names,
            palette=palettes[i],  # Choose a color palette
            legend='full',
            alpha=1,
            s=8,
            marker=markers[i]
        )
        
    plt.legend(
        title='Your Legend Title',
        bbox_to_anchor=(1.05, 1),  # Adjust position to fit outside the plot area if needed
        loc='upper left',
        borderaxespad=0.,
        ncol=4  # Number of columns
    )
        
        
    #plt.legend(markerscale=2, labels=names[np.unique(inverse[indices==i])], ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.title('UMAP Visualization of Sparse Matrix')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_name, dpi=300)


def cache_or_read(plot_name, X, indices=None, harmony=None):
    cached_emb_path = os.path.join('emb_cache', plot_name + 'embedding.pkl')
    if not os.path.exists(cached_emb_path):
        print('cache empty, perform dim reduction...')
        embedding = reduce_dim(X, indices=indices, harmony=harmony)

        with open(cached_emb_path, 'wb') as f:
            pickle.dump(embedding, f)
        
    with open(cached_emb_path, 'rb') as f:
        embedding = pickle.load(f)
    return embedding


class SCMatcher:
    
    def _filter_na_cells(self, cells, atlas_cells, level):
        nan_cells = cells.obs.isna().any(axis=1)
        cells = cells[~nan_cells]
        
        nan_cells = (atlas_cells.obs[level] == 'NA')
        atlas_cells = atlas_cells[~nan_cells]
        
        return cells, atlas_cells
    
    def _filter_common_genes(self, cells, atlas_cells, unify_genes):
        inter_genes = np.intersect1d(cells.var_names, atlas_cells.var_names)
        missing = set(cells.var_names) - set(atlas_cells.var_names)
        if len(missing) > 0:
            missing_str = missing if len(missing) < 100 else str(missing)[:500] + '...'
            
            warning_str = f"{len(missing)} out of {len(cells.var_names)} genes are missing in the Atlas: {missing_str}"
            if not unify_genes:
                warning_str += ". Consider passing unify_genes=True"
            logger.warning(warning_str)
        
        return cells[:, inter_genes], atlas_cells[:, inter_genes]
    
    
    def unify_genes(self, cells, atlas_cells, species):
        logger.info('unifying source gene names')
        cells = unify_gene_names(cells, species)
        logger.info('unifying atlas gene names')
        atlas_cells = unify_gene_names(atlas_cells, species)
        
        return cells, atlas_cells
    
    
    def match_atlas(
        self, 
        name, 
        cells: sc.AnnData, 
        atlas_cells: sc.AnnData, 
        unify_genes=False,
        species='hsapiens',
        level='cell_types',
        plot_atlas=False,
        plot_preds=False,
        **kwargs
    ) -> sc.AnnData:
        
        if unify_genes:
            cells, atlas_cells = self.unify_genes(cells, atlas_cells, species)

        cells, atlas_cells = self._filter_common_genes(cells, atlas_cells, unify_genes)
        
        cells, atlas_cells = self._filter_na_cells(cells, atlas_cells, level)
        
        atlas_cells.obs['cell_types'] = atlas_cells.obs[level]
        
        if plot_atlas:
            embeddings = reduce_dim(atlas_cells.X)
            plot(f'{name}_atlas_plot.jpg', embeddings, atlas_cells.obs['cell_types'].values)
        
        preds = self.match_atlas_imp(name, cells, atlas_cells, level, **kwargs)
        
        if plot_preds:
            embeddings = reduce_dim(cells.X)
            plot(f'{name}_preds_plot.jpg', embeddings, preds.values)
        
        cells.obs['cell_type_pred'] = preds
        
        return cells
        
        
    @abstractmethod
    def match_atlas_imp(
        self, 
        name, 
        cells,
        atlas_cells,
        level
    ):
        """ Output prediction for each cell """
        pass


class HarmonyMatcher(SCMatcher):
    def _prepare_data(self, atlas_cells, cells, sub_atlas, sub_cells, cluster_key, k, random_state):
        import scanpy as sc

        sc.pp.subsample(atlas_cells, fraction=sub_atlas, random_state=random_state)
        sc.pp.subsample(cells, fraction=sub_cells, random_state=random_state)
        assert np.array_equal(cells.to_df().columns, atlas_cells.to_df().columns)
            
        df_combined = pd.concat([cells.to_df(), atlas_cells.to_df()])
        adata_comb = sc.AnnData(df_combined)
        
        if cluster_key is not None:
            logger.info(f"Using custom clusters in .obs[{cluster_key}]")
            cells.obs = cells.obs[['cluster_key']]
        else:
            from sklearn.cluster import KMeans
            logger.info(f"Couldn't find custom clusters, perform kmeans...")
            kmeans = KMeans(n_clusters=k, random_state=random_state).fit(cells.X)
            cells.obs['kmeans_clusters'] = kmeans.labels_.astype(str)
            cells.obs = cells.obs.rename(columns={'kmeans_clusters': 'cell_types'})
            
        adata_comb.obs = pd.concat([cells.obs, atlas_cells.obs])
        adata_comb.obs = pd.DataFrame(adata_comb.obs['cell_types'].values, columns=['cell_types'])
        
        indices = np.concatenate((np.ones(len(cells.obs), dtype=np.int32), np.zeros(len(atlas_cells.obs), dtype=np.int32)))
        return adata_comb, indices
    
    
    def _get_k_nearest(self, X, indices, cell_types, k=5, n=30):
        from sklearn.neighbors import KNeighborsClassifier
        
        xen_emb = X[indices == 1]
        xen_types = cell_types[indices == 1]['cell_types']
        atlas_emb = X[indices == 0]
        atlas_types = cell_types[indices == 0]['cell_types']
        xen_clust_names = np.unique(xen_types)
        logits = []
        model = KNeighborsClassifier(metric='sqeuclidean', n_neighbors=k)
        model.fit(atlas_emb, atlas_types)
        for xen_clust_name in xen_clust_names:    
            xen_emb_i = xen_emb[(xen_types == xen_clust_name).values.flatten(), :]
            idx = np.random.choice(np.arange(len(xen_emb_i)), n)
            xen_emb_i = xen_emb_i[idx]
            

            clusters_pred = model.predict(xen_emb_i)
            names, counts = np.unique(clusters_pred, return_counts=True)
            counts = counts / counts.sum()
            logits_clust = dict(zip(names, counts))
            logits_clust = dict(sorted(logits_clust.items(), key=lambda item: item[1], reverse=True))
            logits.append([xen_clust_name, logits_clust])
            
        logits_df = pd.DataFrame(logits)
        mean_highest = np.array([max(v[1].values()) for v in logits]).mean()
        return logits_df, mean_highest
    
    
    def match_atlas_imp(
        self, 
        name, 
        cells, 
        atlas_cells, 
        sub_atlas=1, 
        sub_cells=0.00005, 
        mode="cells", 
        chunk_len=None, 
        device=None, 
        level='cell_types',
        cluster_key=None,
        k=10,
        random_state=None
    ):
        
        adata_comb, indices = self._prepare_data(atlas_cells, cells, sub_atlas, sub_cells, cluster_key, k, random_state)

        embedding = cache_or_read(name, adata_comb.X, indices=indices, harmony=True)
        
        
        #full_adata_comb, full_indices = self._prepare_data(atlas_cells, cells, sub_atlas=1, sub_cells=1)
        k_nearest, mean_highest = self._get_k_nearest(embedding, indices, adata_comb.obs)
        k_nearest.to_csv(os.path.join('clusters', name + '_k_nearest.csv'))
        cluster_map = k_nearest
        cluster_map['cell_types'] = cluster_map[1].apply(lambda row: max(row, key=row.get))
        cluster_map['cluster'] = cluster_map[0]
        cluster_map = cluster_map[['cluster', 'cell_types']]
        cells.obs['cell_id'] = cells.obs.index 
        #plot_umap(name + 'match_', embedding, adata_comb.obs, indices=indices)
        merged = cells.obs.merge(cluster_map, left_on='cell_types', right_on='cluster', how='left')
        merged.index = merged['cell_id']
        merged = merged['cell_types_y'].values
        return merged
    
class TangramMatcher(SCMatcher):
    
    def match_atlas_imp(
        self, 
        name, 
        cells,
        atlas_cells,
        level, 
        mode="clusters",
        chunk_len=None, 
        device=None,
        random_state=None
    ):
        import scanpy as sc
        import tangram as tg
        import torch

        ad_sp = cells
        ad_sc = atlas_cells

        ad_sc.obs['subclass_label'] = ad_sc.obs[level]

        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if mode == 'cells':
            if chunk_len is None:
                raise ValueError('please provide `chunk_len` when in mode `cells`')
            
            n_chunks = len(cells) // chunk_len
            chunks = []
            for i in range(0, n_chunks + 1):
                start = i * chunk_len
                end = min(i * chunk_len + chunk_len, len(cells))
                chunk = cells[start:end]
                
                tg.pp_adatas(ad_sc, chunk, genes=None)
                
                ad_map = tg.map_cells_to_space(
                                ad_sc, 
                                chunk,
                                num_epochs=200,
                                mode=mode,
                                cluster_label='subclass_label',
                                device=device,
                                random_state=random_state)
                tg.project_cell_annotations(ad_map, chunk, annotation='subclass_label')
                chunks.append(chunk)
            ad_sp = sc.concat(chunks)
            ad_sp.obs.index = cells.obs.index
        else:
            tg.pp_adatas(ad_sc, ad_sp, genes=None)
            
            ad_map = tg.map_cells_to_space(
                            ad_sc, 
                            ad_sp,
                            num_epochs=200,
                            mode=mode,
                            cluster_label='subclass_label',
                            device=device)

            tg.project_cell_annotations(ad_map, ad_sp, annotation='subclass_label')

        preds = pd.Series(ad_sp.obsm['tangram_ct_pred'].idxmax(axis=1), name=level)
        preds.index.name = 'cell_id'
        return preds

    
def matcher_factory(name):
    if name == 'harmony':
        return HarmonyMatcher()
    elif name == 'tangram':
        return TangramMatcher()
    else:
        raise ValueError(f"unknown cell matcher {name}")