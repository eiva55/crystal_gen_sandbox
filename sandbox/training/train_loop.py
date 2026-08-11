"""Training entrypoint for the equivariance-vs-augmentation ablation
(hypothesis 3). Mirrors this repo's existing TensorBoard/Hydra logging
pattern (see run.py) so ablation runs are inspectable the same way as
the five wrapped models' generation runs.

NOT YET FUNCTIONAL — depends on:
  1. Picking the actual backbone architecture (e3nn-based candidate)
  2. Implementing both AblationBackbone variants
  3. Defining the training objective (denoising? reconstruction?)
This file only fixes the entrypoint shape so config wiring can be
tested independently of the architecture decision.
"""
import os
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from sandbox.training.data_module import build_ablation_dataloader


def run_ablation_training(variant: str, dataset_root: str, dataset_limit: int,
                            save_dir: str, max_epochs: int = 10,
                            batch_size: int = 32, **kwargs):
    """
    Args:
        variant: "equivariant" or "augmented" — selects which
            AblationBackbone subclass to train (subclasses TBD).
        dataset_limit: subset size — the independent variable of the
            ablation (sweep this across runs: 1000/5000/20000/45000).
    """
    raise NotImplementedError(
        "Scaffolding only — implement once architecture variants "
        "(sandbox/training/architectures/{equivariant,augmented}.py) exist."
    )
