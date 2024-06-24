# HEST-Library: Bringing Spatial Transcriptomics and Histopathology together
## Designed for querying and assembling HEST-1k dataset 

\[ [arXiv](https://mahmoodlab.github.io/hest-website/) | [HEST-1k](https://huggingface.co/datasets/MahmoodLab/hest) \]
<!-- [ArXiv (stay tuned)]() | [Interactive Demo](http://clam.mahmoodlab.org) | [Cite](#reference) -->

<img src="figures/fig1a.jpg" width="450px" align="right" />

Welcome to the official GitHub repository of the HEST-Library introduced in *"HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis"*. This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. 

<br/>

#### What does this repository provide?
- **HEST-1k:** Free access to <b>HEST-1K</b>, a datasets of 1,108 paired Spatial Transcriptomics samples with HE-stained whole-slide images 
- **HEST-Library:** A series of helpers to assemble new ST samples (from ST, Visium, Visium HD, or Xenium) and work with HEST-1k
- **HEST-Benchmark:** A new benchmark to assess the predicitve performance of foundation models for histology in predicting gene expression from morphology 

<br/>

## Installation

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

## Download/Query HEST-1k

To download/query HEST-1k, follow the tutorial [1-Downloading-HEST-1k](https://github.com/mahmoodlab/HEST/blob/main/tutorials/1-Downloading-HEST-1k.ipynb) or follow instructions on [Hugging Face](https://huggingface.co/datasets/MahmoodLab/hest).

You can then simply view the dataset as, 

```python
from hest import load_hest

print('Lazy loading of hest...')
hest_data = load_hest('hest_data') # location of the data
print('loaded hest')
for d in hest_data:
    print(d)
```

## HEST-Library API

The HEST-Library allows **assembling** new samples using HEST format and **interacting** with HEST-1k. We provide two tutorials:

- [2-Interacting-with-HEST-1k.ipynb](https://github.com/mahmoodlab/HEST/tree/main/tutorials/2-Interacting-with-HEST-1k.ipynb): Playing around with HEST data for loading patches, visualizing nuclear segmentation.
- [3-Assembling-HEST-Data.ipynb](https://github.com/mahmoodlab/HEST/tree/main/tutorials/3-Assembling-HEST-Data.ipynb): Walkthrough to transform a Visum sample into HEST.

In addition, we provide a complete [documentation](https://hest.readthedocs.io/en/latest/).

## HEST-Benchmark

The HEST-Benchmark was designed to assess foundation models for pathology under a new, diverse and challenging benchmark. HEST-Benchmark includes 10 tasks for gene expression prediction (50 highly variable genes) from morphology (112 x 112 um regions at 0.5 um/px) in 10 different organs and 9 cancer types. 

### HEST-Benchmark results (06.24.24)

HEST-Benchmark was used to assess 10 publicly available models. Reported results are based on a Random Forest regression model (70 trees). Additional results based on Ridge regression are provided in the paper. 

|                | **ResNet50** | **KimiaNet** | **Ciga** | **CTransPath** | **Remedis** | **Phikon** | **PLIP** | **UNI** | **CONCH** | **GigaPath** |
|----------------|--------------|--------------|----------|----------------|-------------|------------|----------|---------|-----------|--------------|
| **IDC**        | 0.440        | 0.420        | 0.406    | 0.454          | 0.491       | 0.430      | 0.436    | _0.502_ | **0.504** | 0.492        |
| **PRAD**       | 0.318        | 0.328        | 0.332    | 0.346          | 0.335       | **0.377**  | 0.362    | 0.357   | _0.373_   | 0.372        |
| **PAAD**       | 0.389        | 0.410        | 0.397    | 0.406          | **0.451**   | 0.372      | 0.392    | 0.424   | _0.431_   | 0.425        |
| **SKCM**       | 0.446        | 0.452        | 0.484    | 0.535          | 0.577       | 0.516      | 0.461    | **0.613**| _0.582_   | 0.541        |
| **COAD**       | 0.107        | 0.080        | 0.102    | 0.123          | 0.125       | 0.137      | 0.112    | **0.147**| 0.124     | _0.139_      |
| **READ**       | 0.051        | 0.038        | 0.046    | 0.083          | 0.099       | 0.138      | 0.063    | **0.162**| 0.132     | _0.156_      |
| **CCRCC**      | 0.136        | 0.136        | 0.127    | 0.171          | **0.200**   | 0.178      | 0.124    | _0.186_ | 0.149     | 0.182        |
| **HCC**        | 0.034        | 0.028        | 0.045    | **0.060**      | _0.059_     | 0.041      | 0.038    | 0.051   | 0.040     | 0.055        |
| **LUAD**       | 0.497        | 0.507        | 0.515    | 0.531          | **0.573**   | 0.541      | 0.533    | 0.511   | _0.569_   | 0.547        |
| **LYMPH_IDC**  | 0.205        | 0.206        | 0.218    | 0.238          | 0.243       | 0.243      | 0.229    | 0.234   | **0.249** | _0.248_      |
| **Average**    | 0.262        | 0.261        | 0.267    | 0.295          | 0.315       | 0.297      | 0.275    | **0.319**| 0.315     | _0.316_      |

We provided a step-by-step tutorial to reproduce HEST-Benchmark results in [4-Running-HEST-Benchmark.ipynb](https://github.com/mahmoodlab/HEST/tree/main/tutorials/4-Running-HEST-Benchmark.ipynb)

**Note:** Spontaneous contributions are encouraged if researchers from the community want to include new models. To do so, simply create a Pull Request (see below). 

### Benchmarking your own model

To benchmark your model with hest:

1. Modify the config file in `bench_config/bench_config.yaml`

2. Launch the following:

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

--> move to tutorial 
### Reproducing the results from the paper

To reproduce the results of the HEST-Benchmark (Table 1 and Suppl. Table 11), please follow the following steps:

1. Install HEST-Library as explained in section 1

2. Download the patch encoder weights `fm_v1.zip` from [this link](https://www.dropbox.com/scl/fo/61m7k9s6ujnccdusuv4an/ACcBmaN6LhnluMhDPPGD5fY?rlkey=zqqjxhp7yz0jyrb3ancmo0ofb&dl=0) and unzip it to some directory

3. Then update the paths in the config file `bench_config/bench_config.yaml`

4. Start the benchmark with the following:
```bash
python src/hest/bench/training/predict_expression.py --config bench_config/bench_config.yaml
```

5. Read the results from the `results_dir` specified in the `.yaml` config.

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

## Issues 
- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email `gjaume@bwh.harvard.edu` (and cc `pdoucet@bwh.harvard.edu`). 
- Immediate response to minor issues may not be available.

## Citation

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
