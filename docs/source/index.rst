.. hest documentation master file, created by
   sphinx-quickstart on Sun May  5 11:28:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hest's documentation!
================================

What does the hest library provide?

- Functions for interacting with the `HEST-1K`` dataset
- `Missing file` imputation and automatic alignment for Visium
- `Fast` functions for pooling transcripts and tesselating ST/H&E pairs into patches (these functions are GPU optimized with CUCIM if CUDA is available).
- Deep learning based or Otsu based `tissue segmentation` for both H&E and IHC stains
- Compatibility with `Scanpy`` and `SpatialData`

Hest was used to assemble the HEST-1k dataset, processing challenging ST datasets from a wide variety of sources and converting them to formats commonly used in pathology (.tif, Scanpy AnnData).

The main strength of hest is its ability to read ST samples even when files are missing, for example hest is able to read a Visium sample even if only `filtered_bc_matrix.h5` (or a `mex` folder) and a `full_resolution.tif` are provided.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. automodule:: hest
   :members:
   :undoc-members:
   :show-inheritance:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
