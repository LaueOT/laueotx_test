# LaueOT: a fast optimal transport -based algorithm for indexing polycrystalline sample from Laue experiments

LaueOT finds positions and orientations of grains in polycrystalline samples from Bragg's peaks in white-beam tomographic Laue experiments.
The GPU-based algorithm enables fast analysis of samples with thousands of grains and millions of spots.

## What can LaueOT do?

The inputs to LaueOT are: positions of measured Bragg's peaks (often called spots) on the detector screen for each projectsion, experiment parameters (detector dimensions, distances to the sample, etc). 
The outputs are: a list of grains described by their center in 3D, and orientation matrix with respect to the laboratory reference.
The analysis can take from few minutes on a single CPU for small problems (thousands of spots) to few hours on large GPUs (millions of spots).
Flexible code design enables users to adapt the function to specifics of new problems.

## Installation
Below instructions are meant for Linux/MacOS. On Windows we recommend to use the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

We recommend to use miniconda ([Install Instructions](https://docs.conda.io/en/latest/miniconda.html)) to create a virtual environment for your project. This virtual environment is used to install all required GPU dependencies. 

To get started, first clone this repository
```
git clone https://github.com/LaueOT/laueotx/
cd laueotx
```

Then install all required dependencies
```
conda create -n laueotx -c conda-forge python=3.10 poetry=1.5
conda activate laueotx
conda install -c conda-forge cudatoolkit=11.8.0
poetry install
```

To enable GPU for your conda environment, you need to configure the system paths. You can do it with the following command every time you start a new terminal after activating your conda environment.

```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
```

For your convenience it is recommended that you automate it with the following commands. The system paths will be automatically configured when you activate this conda environment.
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```


If there are issues with the GPU, please follow the [instructions to install tensorflow](https://www.tensorflow.org/install/pip#step-by-step_instructions
)

## Quickstart

For large problems, LaueOT runs in a trivially-parallel mode, by splitting the single-grain fitting step into independent jobs.
Those jobs, called `compute` can be ran at a HPC cluster, for example using Slurm job arrays.
The multi-grain fitting step, called `merge` collects the output of single jobs and produces the model of the final sample.
An example of the process is shown below.

Download the dataset to `tmp` directory (create it first) and run

```
laueotx realdata compute 0 --conf tmp/config_realdata_fega10_v10_demo.yaml -o results/realdata_fega10_v10_demo/ --n-grid 1000
laueotx realdata compute 1 --conf tmp/config_realdata_fega10_v10_demo.yaml -o results/realdata_fega10_v10_demo/ --n-grid 1000
laueotx realdata merge 0 1 --conf tmp/config_realdata_fega10_v10_demo.yaml -o results/realdata_fega10_v10_demo/ --n-grid 1000
```


## Documentation
You can find the [documentation here](https://laueot.github.io/laueotx/)

### Building the docs
The documentation is made using Quarto. To update the documentation you need to [install Quarto](https://quarto.org/docs/get-started/). 
You can render the new website (locally) using the following commands (make sure to have laueotx installed properly):
```
conda activate laueotx
cd docs/
quarto render
```


The webpage can be published to github pages using the following command. It will also be re-rendered in the process.
```
quarto publish gh-pages
```
