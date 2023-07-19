

## Installation

We recommend to use miniconda to create a virtual environment for your project. This virtual environment is used to install all required GPU dependencies.

```
conda create -n laueotx -c conda-forge python=3.10 poetry=1.5
conda activate laueotx
conda install -c conda-forge cudatoolkit=11.8.0
poetry install
```

Enable GPU for your conda environment
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda activate laueotx
```

If there are issues with the GPU, please follow the [instructions to install tensorflow](https://www.tensorflow.org/install/pip#step-by-step_instructions
)

## Quickstart

Download the dataset to `tmp` directory (create it first) and run

```
laueotx realdata compute 0 --conf tmp/config_realdata_fega10_v10_demo.yaml -o results/realdata_fega10_v10_demo/ --n-grid 1000
laueotx realdata compute 1 --conf tmp/config_realdata_fega10_v10_demo.yaml -o results/realdata_fega10_v10_demo/ --n-grid 1000
laueotx realdata merge 0 1 --conf tmp/config_realdata_fega10_v10_demo.yaml -o results/realdata_fega10_v10_demo/ --n-grid 1000
```
