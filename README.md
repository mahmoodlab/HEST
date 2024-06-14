# HEST-LIB

\[ [HEST-1k dataset (incoming)](https://huggingface.co/datasets/MahmoodLab/hest) | [website](https://mahmoodlab.github.io/hest-website/)\]
<!-- [ArXiv (stay tuned)]() | [Interactive Demo](http://clam.mahmoodlab.org) | [Cite](#reference) -->

**Note: HEST is still under review and in active development. Please report any bugs in the GitHub issues (The full HEST-1k dataset will be made available soon. Stay tuned)** 
<br/>


#### What does the hest library provide?
- Functions for interacting with the <b>HEST-1K</b> dataset
- <b>Missing file</b> imputation and automatic alignment for Visium
- <b>Fast</b> functions for pooling transcripts and tesselating ST/H&E pairs into patches (these functions are GPU optimized with CUCIM if CUDA is available).
- Deep learning based or Otsu based <b>tissue segmentation</b> for both H&E and IHC stains
- Compatibility with <b>Scanpy</b> and <b>SpatialData</b>

Hest was used to assemble the HEST-1k dataset, processing challenging ST datasets from a wide variety of sources and converting them to formats commonly used in pathology (.tif, Scanpy AnnData).

<p align="center">
  <img src="figures/fig1.png" alt="Description" style="width: 70%;"/>
</p>

The main strength of hest is its ability to read ST samples even when files are missing, for example hest is able to read a Visium sample even if only `filtered_bc_matrix.h5` (or a `mex` folder) and a `full_resolution.tif` are provided.

<br/>

1. [Installation](#installation)
2. [Information for reviewers](#information-for-reviewers)
3. [Documentation](https://hest.readthedocs.io/en/latest/)
3. [Query HEST-1k](#downloadquery-hest-1k)
4. [Hest-lib tutorials](#tutorials) \
5.2 [HESTData](#hestdata-api) \
5.2 [Hest reader API](#hest-reader-api) \
5.3 [Hest bench](#hest-bench-tutorial) 
<br/>


# Installation

```
git clone https://github.com/mahmoodlab/hest.git
cd hest
conda create -n "hest" python=3.9
conda activate hest
pip install -e .
```

#### additional dependencies (for WSI manipulation):
```
sudo apt install libvips libvips-dev openslide-tools
```

#### additional dependencies (GPU acceleration):
If a GPU is available on your machine, it is strongly recommended to install [cucim](https://docs.rapids.ai/install) on your conda environment. (hest was tested with `cucim-cu12==24.4.0` and `CUDA 12.1`)
```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.* \
    raft-dask-cu12==24.6.*
```

NOTE: hest was only tested on linux/macOS machines, please report any bugs in the GitHub issues.

### Tissue segmentation model
In order to use the deep-learning based tissue segmentation model:
1. Download the segmenter weights [here](https://huggingface.co/pauldoucet/tissue-detector/blob/main/deeplabv3_seg_v4.ckpt)
2. place the weights in the `models/` directory

### CONCH/UNI installation (Optional, for HEST-bench only)

If you want to benchmark CONCH/UNI, additional steps are necesary

#### CONCH installation (model + weights)

1. Request access to the model weights from the Huggingface model page [here](https://huggingface.co/MahmoodLab/CONCH).

2. Download the model weights (`pytorch_model.bin`) and place them in your `fm_v1` directory `fm_v1/conch_v1_official/pytorch_model.bin`

3. Install the CONCH PyTorch model:

```
git clone https://github.com/mahmoodlab/CONCH.git
cd CONCH
pip install -e .
```

#### UNI installation (weights only)

1. Request access to the model weights from the Huggingface model page [here](https://huggingface.co/MahmoodLab/UNI).

2. Download the model weights (`pytorch_model.bin`) and place them in your `fm_v1` directory `fm_v1/uni_v1_official/pytorch_model.bin`

# Information for reviewers


In order to reproduce the results of the HEST-benchmark (Table 1 and Suppl. Table 11), please follow the following steps:

1. Install hest as explained in section 1

2. Download the benchmark task data `bench_data.zip` from [this link](https://www.dropbox.com/scl/fo/61m7k9s6ujnccdusuv4an/ACcBmaN6LhnluMhDPPGD5fY?rlkey=zqqjxhp7yz0jyrb3ancmo0ofb&dl=0) and unzip it to some directory

3. Download the patch encoder weights `fm_v1.zip` from [this link](https://www.dropbox.com/scl/fo/61m7k9s6ujnccdusuv4an/ACcBmaN6LhnluMhDPPGD5fY?rlkey=zqqjxhp7yz0jyrb3ancmo0ofb&dl=0) and unzip it to some directory

4. Then update the paths in the config file `bench_config/bench_config.yaml`

5. Start the benchmark with the following:
```bash
python src/hest/bench/training/predict_expression.py --config bench_config/bench_config.yaml
```

6. Read the results from the `results_dir` specified in the `.yaml` config.


# Download/Query HEST-1k

In order to download/query HEST-1k, please follow [this tutorial](https://huggingface.co/datasets/MahmoodLab/hest).
The data will be in open access soon!

# Tutorials

## HESTData API

## Read a HESTData sample from disk
```python
st = read_HESTData(
    adata_path='SPA154.h5ad', # aligned ST counts
    img='SPA154.tif', # WSI
    metrics_path='SPA154.json', # metrics/metadata
    mask_path_pkl='SPA154_mask.pkl', # optional (tissue_mask)
    mask_path_jpg='SPA154_mask.jpg' # optional (tissue_mask)
)
```

## Visualizing the spots over a fullres WSI
``` python
# visualize the spots over a downscaled version of the fullres image
st.save_spatial_plot(save_dir)
```

## Saving to pyramidal tiff and h5
Save `HESTData` object to `.tiff` + expression `.h5ad` and a metadata file.
``` python
# Warning saving a large image to pyramidal tiff (>1GB) can be slow on a hard drive !
st.save(save_dir, pyramidal=True)

```

## Otsu-based segmentation or deep learning based segmentation

In order to use the deep-learning based detector, please download the weights from [here](https://huggingface.co/pauldoucet/tissue-detector), and deposit then in the `models/` directory.

```python
save_dir = '.'

st.compute_mask(method='deep') # or method='otsu'
st.save_tissue_seg_pkl(save_dir, name)
st.save_tissue_seg_jpg(save_dir, name)
```

## Patching

```python

st.dump_patches(
    patch_save_dir,
    'demo',
    target_patch_size=224,
    target_pixel_size=0.5
)
```

## Hest reader API

## Reading legacy Visium files

### When should I provide an alignment file and when should I use the autoalignment?

#### Step 1: check if a tissue_positions.csv/tissue_position_list.csv already provides a correct alignment
Most of the time if a spatial/ folder containing a tissue_positions.csv/tissue_position_list.csv is available you don't need any autoalignment/alignment file. Try the following: `st = VisiumReader().read(fullres_img_path, bc_matric_path, spatial_coord_path=spatial_path)` (where `spatial_path` is a folder that contains a tissue_positions.csv or a tissue_position_list.csv), you can then double check the alignment (`st.save_spatial_plot(save_dir)`) by saving a visualization plot that takes the full resolution image, downscale it and overlays it with the spots. If the alignment looks off, try step 2.

#### Step 2: check if a .json alignment file is provided
If a .json alignment file is available, try the following `VisiumReader().read(fullres_img_path, bc_matric_path, spatial_coord_path=spatial_path, alignment_file_path=align_path)` you can also ommit the spatial_coord_path `VisiumReader().read(fullres_img_path, bc_matric_path, alignment_file_path=align_path)`


#### Step 3: attempt auto-alignment
If at least 3 corner fiducials are not cropped out and are reasonably visible, you can attempt an autoalignment with `VisiumReader().read(fullres_img_path, bc_matric_path`. (if no spatial folder and no alignment_file_path is provided, it will attempt autoalignment by default, you can always force auto-alignment by passing `autoalign='always'`)


### Reading from a filtered_feature_bc_matrix.h5, an image and a spatial/ folder
```python
from hest import VisiumReader

st = VisiumReader().read(
    fullres_img_path, # path to a full res image
    bc_matric_path, # path to filtered_feature_bc_matrix.h5
    spatial_coord_path=spatial_coord_path # path to a space ranger spatial/ folder containing either a tissue_positions.csv or tissue_position_list.csv
)

# if no spatial folder is provided, but you have an alignment file
st = VisiumReader().read(
    fullres_img_path, # path to a full res image
    bc_matric_path, # path to filtered_feature_bc_matrix.h5
    alignment_file_path=alignment_file_path # path to a .json alignment file
)

# if both the alignment file and the spatial folder are missing, attempt autoalignment
st = VisiumReader().read(
    fullres_img_path, # path to a full res image
    bc_matric_path, # path to filtered_feature_bc_matrix.h5
)

```

### Auto read
Given that `visium_dir` contains a full resolution image and all the necessary Visium files such as the `filtered_bc_matrix.h5` and the `spatial` folder, `VisiumReader.auto_read(path)` should be able to automatically read the sample. Prefer `read` for a more fine grain control.

```python
from hest import VisiumReader

visium_dir = ...

# attempt autoread
st = VisiumReader().auto_read(visium_dir)
```

## Hest-bench tutorial

In order to benchmark your model with hest

1. Download the benchmark task data `bench_data.zip` from [this link](https://drive.google.com/drive/folders/1x5envjv6lUfH9Hw13hXPIucMJELSJEKl) and unzip it to some directory

2. Then modify the config file in `bench_config/bench_config.yaml`

### Benchmarking your own model

```python
from hest.bench import benchmark_encoder

PATH_TO_CONFIG = .. # path to `bench_config.yaml`
model = .. # PyTorch model (torch.nn.Module)
model_transforms = .. # transforms to apply during inference (torchvision.transforms.Compose)

benchmark_encoder(        
    model, 
    model_transforms,
    PATH_TO_CONFIG
)
```

### From the command-line

1. Add your model config in `src/hest/bench/cpath_model_zoo/pretrained_configs`
2. Launch the benchmark with:

```
python src/hest/bench/training/predict_expression.py --config bench_config/bench_config.yaml
```