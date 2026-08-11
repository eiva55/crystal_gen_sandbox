"""Wraps the existing MP20Dataset for the equivariance-vs-augmentation
ablation (hypothesis 3): needs the SAME underlying data across both
architecture variants, varied only by `limit` (subset size), so the
comparison isolates dataset scale as the independent variable.

Reuses sandbox.datasets.mp20.MP20Dataset as-is — no need for a separate
loader, this ablation doesn't need anything MP20Dataset doesn't already do.
"""
from torch.utils.data import DataLoader
from sandbox.datasets.mp20 import MP20Dataset


def build_ablation_dataloader(root: str, split: str, limit: int,
                                batch_size: int = 32, num_workers: int = 0,
                                shuffle: bool = True) -> DataLoader:
    dataset = MP20Dataset(root=root, split=split, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, collate_fn=lambda x: x)
    # collate_fn=identity: pymatgen.Structure objects, not tensors — actual
    # tensorization happens inside the model's forward pass depending on
    # which architecture variant (equivariant vs augmented) consumes them.
