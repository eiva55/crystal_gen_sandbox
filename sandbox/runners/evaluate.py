import os
import torch
from typing import List, Optional
from pymatgen.core import Structure
from sandbox.metrics.crystal_metrics import CrystalMetrics


def evaluate_generation(model, num_samples: int, batch_size: int, device: torch.device,
                         save_dir: str = None, dataset=None, task=None, viz_enabled: bool = False, **kwargs):
    """Generate structures and compute validity/uniqueness/novelty against dataset."""
    model.to(device)
    structures = model.generate(num_samples, batch_size, device, save_dir=save_dir)
    if not structures:
        print("No structures generated.")
        return {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, s in enumerate(structures):
            if s is not None:
                s.to(os.path.join(save_dir, f"gen_{i}.cif"))

    if viz_enabled and save_dir and task is not None:
        task.visualize(structures, save_dir)

    reference_structures = list(dataset) if dataset is not None else None
    metrics = CrystalMetrics.compute_all(structures, reference_structures)
    print("Metrics:", metrics)
    return metrics
