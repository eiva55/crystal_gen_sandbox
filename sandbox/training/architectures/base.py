"""Shared interface for the equivariance-vs-augmentation ablation
(hypothesis 3). Both variants below must expose the same forward
signature so the training loop is architecture-agnostic — the only
difference between them is whether SE(3)-equivariance is architecturally
enforced or approximated via data augmentation.
"""
from abc import ABC, abstractmethod
import torch.nn as nn


class AblationBackbone(nn.Module, ABC):
    """Common contract for both ablation variants.

    forward() takes a batch of pymatgen Structures (or a pre-tensorized
    batch — TBD once the actual architecture is picked) and returns
    whatever the training objective needs (e.g. denoising targets, if
    following a diffusion-style objective like the rest of this repo's
    models, or reconstruction loss components — architecture decision
    pending).
    """

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_equivariant_by_construction(self) -> bool:
        """True for the explicit-equivariance variant, False for the
        augmentation-only variant. Used for logging/bookkeeping, not
        control flow."""
        raise NotImplementedError
