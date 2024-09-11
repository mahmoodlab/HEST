# API


## Interact with HEST-1k

```{eval-rst}
.. module:: hest
```

```{eval-rst}
.. currentmodule:: hest.HESTData

.. autosummary::
    :toctree: generated
   
    load_hest
```

## Run HEST-Benchmark

```{eval-rst}
.. module:: hest
```

```{eval-rst}
.. currentmodule:: hest.bench

.. autosummary::
    :toctree: generated
   
    benchmark_encoder
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

## Resolving gene name aliases

```{eval-rst}
.. currentmodule:: hest.HESTData

.. autosummary::
    :toctree: generated

    unify_gene_names
```


## Readers to augment HEST-1k

Readers to create new HEST-1k samples.

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
Nuclei segmentation methods


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