## Space Group Constrained Crystal Generation [ICLR 2024]

Implementation codes for Space Group Constrained Crystal Generation (DiffCSP++).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jiaor17/DiffCSP-PP/blob/main/LICENSE)   [**[Paper]**](https://openreview.net/pdf?id=jkvZ7v4OmP)

![Overview](assets/overview.png "Overview")

### Setup

Complete `.env` with the following correctly filled out
```
PROJECT_ROOT="${THIS_FOLDER}"
HYDRA_JOBS="${THIS_FOLDER}/results"
WANDB_DIR="${THIS_FOLDER}"
USE_WANDB_LOGGING=1
```

One can specifiy `USE_WANDB_LOGGING=0` to disable wandb.

```
micromamba env create -f environment.yml
micromamba activate diffcsp_pp
```

### Training

For the CSP task

```
python diffcsp/run.py data=<dataset> expname=<expname>
```

For the Ab Initio Generation task

```
python diffcsp/run.py data=<dataset> model=diffusion_w_type expname=<expname>
```

The ``<dataset>`` tag can be selected from perov_5, mp_20, mpts_52 and carbon_24. Support for the alex_mp_20 dataset can be added by following the extension instructions available in this [repository](https://github.com/omri1348/SGFM/blob/main/data/alex_mp_20/README.md). Pre-trained checkpoints are provided [here](https://drive.google.com/drive/folders/1FQ_b6CE09KtyGaU_r6uO8_I5JhrQmUFB?usp=sharing).

### Evaluation

#### Crystal Structure Prediction

```
python scripts/evaluate.py --model_path <model_path> --dataset <dataset> # with GT Wyckoff positions
python script/csp_from_template.py --model_path <csp model> --csv_path <path containing *_comp_fps.csv> --finder_model_path <CSPML model> # with CSPML templates
python scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv 
```

#### Ab initio generation

```
python scripts/generation.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics.py --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
```

### Sampling APIs for pre-defined symmetries

One can sample structures given the symmetry information by the following codes:

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --spacegroup <spacegroup> --wyckoff_letters <wyckoff_letters> --atom_types <atom_types>
```

One example can be

```
python scripts/sample.py --model_path mp_csp --save_path MnLiO --spacegroup 58 --wyckoff_letters 2a,2d,4g --atom_types Mn,Li,O
# Or simplify WPs as
python scripts/sample.py --model_path mp_csp --save_path MnLiO --spacegroup 58 --wyckoff_letters adg --atom_types Mn,Li,O
```

If multiple structures are required to be generated, one can first collect the inputs into a json file like `example/example.json`:

```
[
    {
        "spacegroup_number": 58,
        "wyckoff_letters": ["2a","2d","4g"],
        "atom_types": ["Mn","Li","O"]
    },
    {
        "spacegroup_number": 194,
        "wyckoff_letters": "abff",
        "atom_types": ["Tm","Tm","Ni","As"]
    }
]
```

And parallelly generate the structures via

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --json_file <json_file>
```

Notably, the `atom_types` parameter is not required if a model for ab initio generation is used.


## Acknowledgments

This codebase is mainly built upon [CDVAE](https://github.com/txie-93/cdvae) and [DiffCSP](https://github.com/jiaor17/DiffCSP). For the datasets, Perov-5, Carbon-24 and MP-20 are from [CDVAE](https://github.com/txie-93/cdvae), and MPTS-52 is collected from its original [codebase](https://github.com/sparks-baird/mp-time-split).

## Citation

Please consider citing our work if you find it helpful:

```
@inproceedings{
jiao2024space,
title={Space Group Constrained Crystal Generation},
author={Rui Jiao and Wenbing Huang and Yu Liu and Deli Zhao and Yang Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=jkvZ7v4OmP}
}
```

## Contact

If you have any questions, feel free to reach us at:

Rui Jiao: [jiaor21@mails.tsinghua.edu.cn](mailto:jiaor21@mails.tsinghua.edu.cn)
