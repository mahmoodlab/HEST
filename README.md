# HEST-Library: Bringing Spatial Transcriptomics and Histopathology together
## Designed for querying and assembling HEST-1k dataset 

\[ [HEST-1k (incoming)](https://huggingface.co/datasets/MahmoodLab/hest) | [website](https://mahmoodlab.github.io/hest-website/)\]
<!-- [ArXiv (stay tuned)]() | [Interactive Demo](http://clam.mahmoodlab.org) | [Cite](#reference) -->

<img src="figures/fig1a.jpg" width="450px" align="right" />

Welcome to the official GitHub repository of the HEST-Library introduced in *"HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis"*. This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. 

**Note: HEST-Library is still under active development. Please report any bugs in the GitHub issues.** 
<br/>

#### What does the HEST-Library provide?
- Downloading <b>HEST-1K</b>, the largest dataset of paired Spatial Transcriptomics and Histology
- A series of helpers to unify ST, Visium, Visium HD, and Xenium data, e.g., automatic ST/WSI alignment
- Running the HEST-Benchmark for assessing foundation models for histology

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
If a GPU is available on your machine, we recommend installing [cucim](https://docs.rapids.ai/install) on your conda environment. (hest was tested with `cucim-cu12==24.4.0` and `CUDA 12.1`)
```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.* \
    raft-dask-cu12==24.6.*
```

NOTE: HEST-Library was only tested on Linux/macOS machines, please report any bugs in the GitHub issues.

# Download/Query HEST-1k

To download/query HEST-1k, follow the instructions on the [Hugging Face page](https://huggingface.co/datasets/MahmoodLab/hest). The data will be in open access soon!

You can then simply load the dataset as a `List[HESTData]`
```python
from hest import load_hest

print('load hest...')
hest_d = load_hest('hest_data') # location of the data
print('loaded hest')
for d in data:
    print(d)
```

# HEST-Library API

To see concrete examples of the HEST-library API, refer to the example notebook in `examples/tutorial.ipynb`.

You can also visit the HEST-library [documentation](https://hest.readthedocs.io/en/latest/).

# HEST-Benchmark

To reproduce the results of the HEST-Benchmark (Table 1 and Suppl. Table 11), please follow the following steps:

1. Install HEST-Library as explained in section 1

2. Download the benchmark task data `bench_data.zip` from [this link](https://www.dropbox.com/scl/fo/61m7k9s6ujnccdusuv4an/ACcBmaN6LhnluMhDPPGD5fY?rlkey=zqqjxhp7yz0jyrb3ancmo0ofb&dl=0) and unzip it to some directory

3. Download the patch encoder weights `fm_v1.zip` from [this link](https://www.dropbox.com/scl/fo/61m7k9s6ujnccdusuv4an/ACcBmaN6LhnluMhDPPGD5fY?rlkey=zqqjxhp7yz0jyrb3ancmo0ofb&dl=0) and unzip it to some directory

4. Then update the paths in the config file `bench_config/bench_config.yaml`

5. Start the benchmark with the following:
```bash
python src/hest/bench/training/predict_expression.py --config bench_config/bench_config.yaml
```

6. Read the results from the `results_dir` specified in the `.yaml` config.

### CONCH/UNI installation (Optional, for HEST-Benchmark only)

If you want to benchmark CONCH/UNI, additional steps are necessary

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

### Benchmarking your own model

To benchmark your model with hest:

1. Download the benchmark task data `bench_data.zip` from [this link](https://drive.google.com/drive/folders/1x5envjv6lUfH9Hw13hXPIucMJELSJEKl) and unzip it to some directory

2. Then modify the config file in `bench_config/bench_config.yaml`

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

#### From the command-line

1. Add your model config in `src/hest/bench/cpath_model_zoo/pretrained_configs`
2. Launch the benchmark with:

```
python src/hest/bench/training/predict_expression.py --config bench_config/bench_config.yaml
```

# Issues 
- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email `gjaume@bwh.harvard.edu` (and cc `pdoucet@bwh.harvard.edu`). 
- Immediate response to minor issues may not be available.

# Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{jaume2024hest,
  title={HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis},
  author={Jaume, Guillaume and Doucet, Paul and Song, Andrew H. and Lu, Ming Y. and Almagro-Pérez, Cristina and Wagner, Sophia J. and Vaidya, Anurag J. and Chen, Richard J.and Williamson, Drew F.K. and Kim, Ahrong and Mahmood, Faisal},
  booktitle={arXiv},
  year={2024}
}
```

<img src=docs/joint_logo.png> 
