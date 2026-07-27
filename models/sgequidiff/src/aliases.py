import argparse
import os
from pathlib import Path
from typing import *

import numpy
import omegaconf
import torch
import pyxtal
import pymatgen
import torch_geometric

# --- string types
path_t: type = Union[os.PathLike, str]

# --- numeric types
Tensor: type = torch.Tensor
tensor: type = Tensor  # TODO deprecate
Distribution: type = torch.distributions.Distribution
Module: type = torch.nn.Module
Optimizer: type = torch.optim.Optimizer
LRScheduler: type = torch.optim.lr_scheduler.LRScheduler
Parameter: type = torch.nn.Parameter
ndarray: type = numpy.ndarray

# --- configuration types
namespace: type = argparse.Namespace
DictConfig: type = omegaconf.DictConfig

# --- object types
group: type = pyxtal.symmetry.Group
pmg_structure: type = pymatgen.core.structure.Structure
pyg_graph: type = torch_geometric.data.Data
pyg_graph_batch: type = torch_geometric.data.batch.Batch
