

## Quickstart

We recommend to use miniconda to create a virtual environment for your project. This virtual environment is used to install all required GPU dependencies.


https://www.tensorflow.org/install/pip#step-by-step_instructions


```
conda create -n laueotx -c conda-forge python=3.10 poetry=1.5
conda install -c conda-forge cudatoolkit=11.8.0 poetry
conda activate laueotx
poetry install
```


Download the dataset to `tmp` directory (create it first) and run

```
laueotx realdata compute 0 --conf tmp/config_realdata_fega10_v10_demo.yaml -o results/realdata_fega10_v10_demo/ --n-grid 1000
```
