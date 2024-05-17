# ST-histology-loader
hest provides a flexible approach for loading the different Spatial Transcriptomics data formats supporting H&E (Visium/Visium-HD, Xenium and ST) and for automatically aligning them with their associated histology image. Hest was used for assembling the HEST-1k dataset, processing challenging ST datasets from a wide variety of sources.


# Installation

```
conda create -n "hest" python=3.9
conda activate hest
pip install -e .
```


# HEST-bench tutorial

In order to use the HEST-benchmark, start by placing all the hest-formated datasets in a common folder.

Then modify the config file in samples/bench_config.yaml

Finally launch the benchmark with:
```
python src/hest/bench/training/predict_expression.py --config samples/bench_config.yaml
```