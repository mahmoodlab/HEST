# API


## Interact with HEST-1k

```{eval-rst}
.. module:: hest
```

```{eval-rst}
.. currentmodule:: hest.HESTData

.. autosummary::
    :toctree: generated
   
    iter_hest
```

## Run HEST-Benchmark

```{eval-rst}
.. module:: hest.bench

.. autosummary::
    :toctree: generated
   
    benchmark
```

## HESTData class

```{eval-rst}
.. module:: hest
```

```{eval-rst}
.. currentmodule:: hest.HESTData

.. autosummary::
    :toctree: generated
   
    HESTData
```

## Pooling of transcripts, binning

Methods used to pool Xenium transcripts and Visium-HD bins into square bins of custom size

```{eval-rst}
.. currentmodule:: hest.readers

.. autosummary::
    :toctree: generated
   
    pool_transcripts_xenium
    pool_bins_visiumhd
```

## Batch effect visualization/correction

```{eval-rst}
.. module:: hest
```

```{eval-rst}
.. currentmodule:: hest.batch_effect

.. autosummary::
    :toctree: generated
   
    filter_hest_stromal_housekeeping
    get_silhouette_score
    plot_umap
    correct_batch_effect
```

## Gene names manipulation

```{eval-rst}
.. currentmodule:: hest.HESTData

.. autosummary::
    :toctree: generated

    unify_gene_names
    ensembl_id_to_gene
```


## Readers to expand HEST-1k

Readers to expand HEST-1k with additional samples.

```{eval-rst}
.. currentmodule:: hest.readers

.. autosummary::
    :toctree: generated

    Reader
    VisiumReader
    XeniumReader
    VisiumHDReader
    STReader
```


## CellViT segmentation
Simplified API for nuclei segmentation


```{eval-rst}
.. currentmodule:: hest.segmentation.cell_segmenters

.. autosummary::
    :toctree: generated

    segment_cellvit
```


## Miscellaneous

```{eval-rst}
.. currentmodule:: hest

.. autosummary::
    :toctree: generated

    tiff_save
    autoalign_visium
    write_10X_h5
    find_pixel_size_from_spot_coords
```