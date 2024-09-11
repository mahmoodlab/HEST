# Installing `hest`

Simply clone and install the package as follows:
```
git clone https://github.com/mahmoodlab/HEST.git
cd HEST
conda create -n "hest" python=3.9
conda activate hest
pip install -e .
```

## Additional dependencies (for WSI manipulation):

```
sudo apt install libvips libvips-dev openslide-tools
```

## Install CuImage for GPU acceleration

For `CuImage` support (GPU accelerated library), follow the instructions provided by Nvidia.

Example for `Cuda 12.1`:

```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.* \
    raft-dask-cu12==24.6.*
```

**NOTE:** HEST-Library was only tested on Linux/macOS machines, please report any bugs in the GitHub issues.