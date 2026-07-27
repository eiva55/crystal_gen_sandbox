# SGEquiDiff

<h4 align="left">

[![arXiv](https://img.shields.io/badge/arXiv-2505.10994-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2505.10994)

</h4>

This repository hosts the official implementation of Space Group Equivariant 
Crystal Diffusion (SGEquiDiff).

<p align="center">
  <img src="assets/fig_overview.png" /> 
</p>

<p align="center">
  <img src="assets/sgequidiff_animation.gif" width="200">
</p>

## Table of Contents
- [Installation](#installation)
- [Pre-processing](#pre-processing)
- [Training](#training)
- [Generation](#generation)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Installation
> Note that the following installation instructions assume that you have a CUDA
> GPU.

### Installation with `uv`
> Note that the following assumes you are on a Linux operating system.

Run the following to install everything with [uv](https://docs.astral.sh/uv/):
```
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install "setuptools<81" wheel
uv pip install -e .
```

### Docker
Obtain a Docker installation for free [here](https://docs.docker.com/desktop/?_gl=1*zxjt97*_ga*Nzc1ODQ3ODExLjE2ODQxMTEzODU.*_ga_XJWPQMJYHQ*MTcwMzE5Nzk5NS40Mi4xLjE3MDMxOTgyMTYuNTUuMC4w) and follow their instructions. 
You can make sure Docker is running on your system using the `docker version` 
command from your shell. We provide a default `Dockerfile` containing the 
environment configuration based on NVIDIA CUDA 12.4.0.41 (see [here](https://docs.nvidia.com/deeplearning/frameworks/pyg-release-notes/rel-24-03.html#rel-24-03)).
To build the Docker image, navigate to the project directory and run

```
docker build --tag sgequidiff:latest .
```

To start a new container from the image, you can use
```
docker run -dt -v "$(pwd)":/home/sgequidiff --name sgequidiff sgequidiff:latest /bin/bash
```
If the container already exists, you can start it with
```
docker start sgequidiff
```
To open an interactive bash shell in the container, you can use
```
docker exec -it sgequidiff /bin/bash
```

#### Apptainer/Singularity
Many shared clusters do not allow users to execute Docker commands for security
purposes. In this case, we can copy our Docker image to the cluster and build it
there with Apptainer (formerly Singularity).

Run `docker images` and get the image hash from the `IMAGE_ID` column. Compress
the image and copy it to your cluster with the following:

```
docker save <hash here> -o sgequidiff_image.tar
rsync -P sgequidiff_image.tar <destination here>
```

On your cluster, navigate to the `.tar` and build a `.sif` file with

```
apptainer build sgequidiff.sif docker-archive://sgequidiff_image.tar
```

Put the `.sif` file wherever you want, but we will assume you place it in the 
project directory.

## Environment variables

Before running any scripts, set the following environment variables:
```
export PROJECT_ROOT=<path to this repo's project directory>
export HYDRA_JOBS=<path to a folder to store hydra outputs>
export WANDB_DIR=<path to folder to store wandb outputs>
export WANDB_API_KEY=<insert your wandb API key here>
```

If not installed with Docker, you will additionally need to set the following
environment variable (our Dockerfile takes care of this):
```
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}/src
```

If using Apptainer, prepend the following to all Python 
executables:

```
apptainer run -e --nv --bind ${PROJECT_ROOT}:/home/sgequidiff ./sgequidiff.sif
```

## Pre-processing

Pre-processed data can be downloaded online for MP20 [here](https://drive.google.com/file/d/1avOCeo-OMtkKYPkDO0pz0IY36gBq8-kE/view?usp=sharing)
and for MPTS52 [here](https://drive.google.com/file/d/1rPMi0HKMtBccAczkgg-S-kCqSS63Jy1T/view?usp=sharing).
Unzip and place the `mp_20` and `mpts_52` folders inside the `SGEquiDiff/data` 
directory.

Alternatively, the datasets (as CIFs) can be pre-processed manually into 
asymmetric unit objects with the following command:
```
python scripts/preprocess_datasets.py --dataset <dataset_name>
```
Replace `<dataset_name>` with `mp20` or `mpts52`. On a 2020 Macbook Pro, 
pre-processing takes approximately 20 minutes. The script will output a series
of warnings from the `spglib` package, but these can be ignored.

## Training
Model checkpoints can be found [here](https://drive.google.com/drive/folders/1ONwO53i6oG1_yBqP0zPQ-IV_zLoIWyQR?usp=sharing). To train models yourself, training can be executed as follows:

```
python experiments/run.py dataset=<dataset_name> wandb.entity=<your entity>
```

Make sure to replace `<dataset_name>` with `mp_20` or `mpts_52` and 
`<your entity>` with your Weights and Biases entity. Once training has finished,
module checkpoints will be saved at the following path, which we refer to as 
`MODEL_PATH`:
```
${PROJECT_ROOT}/experiment_logs/generative_crystals/<some foldername starting with the date>
```

## Generation

Generation can be executed as follows:

```
python scripts/generate_crystals.py --num_samples 10_000 --batch_size 500 --ckpt_dir <MODEL_PATH> --load_best_submodules --temperature <temperature>
```

Replace `<MODEL_PATH>` and `<temperature>` as desired. A `.npz` of flattened
crystals will be generated inside `MODEL_PATH`. We refer to the `.npz` filepath
as `XTALS_PATH`.

## Evaluation

Run the following to compute proxy metrics:

```
python scripts/evaluate_samples.py --dataset_name <dataset_name> --gen_crystals_path <XTALS_PATH>
```

Once finished, metrics will be printed and saved as a json file to the folder
enclosing `XTALS_PATH`.

## Crystal structure prediction task

For the LDiff variant of SGEquiDiff and evaluation on the crystal structure
prediction task from the paper, switch to the `latdiff` branch and see 
the `README.md`.

## Citation

Please consider citing the following paper if you find our code useful.

```
@misc{chang2025spacegroupequivariantcrystal,
      title={Space Group Equivariant Crystal Diffusion}, 
      author={Rees Chang and Angela Pak and Alex Guerra and Ni Zhan and Nick Richardson and Elif Ertekin and Ryan P. Adams},
      year={2025},
      eprint={2505.10994},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2505.10994}, 
}
```
