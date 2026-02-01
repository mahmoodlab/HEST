# API


## Interact with HEST-1k

See tutorial `2. Interacting with HEST`.

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

See tutorial `4. Running HEST Benchmark`.

```{eval-rst}
.. module:: hest.bench

.. autosummary::
    :toctree: generated
   
    benchmark
```

## HESTData class

Core object representing a (pooled) Spatial Transcriptomics sample along with a full resolution H&E image and associated metadata. See tutorial `2. Interacting with HEST`.

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
    pool_bins_visiumhd_per_cell
```

## CellViT segmentation
Simplified API for nuclei segmentation


```{eval-rst}
.. currentmodule:: hest.segmentation.cell_segmenters

.. autosummary::
    :toctree: generated

    segment_cellvit
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

Readers to expand HEST-1k with additional samples. See tutorial `3. Assembling HEST Data`.

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

## IO

```{eval-rst}
.. currentmodule:: hest.io.seg_readers

.. autosummary::
    :toctree: generated

    GDFReader
    XeniumParquetCellReader
    GDFParquetCellReader
    XeniumTranscriptsReader
    HESTXeniumTranscriptsReader
    write_geojson
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


## Miscellaneous

```{eval-rst}
.. currentmodule:: hest

.. autosummary::
    :toctree: generated

    tiff_save
    autoalign_visium
    write_10X_h5
    find_pixel_size_from_spot_coords
    get_k_genes
```