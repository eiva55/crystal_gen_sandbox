from sandbox.contracts.base import BaseTask
import torch
from typing import Any, List

class CrystalGenerationTask(BaseTask):
    def run(self, model, num_samples, batch_size, device, save_dir=None, **kwargs):
        print(f"Running generation task with model {model.__class__.__name__}")
        structures = model.generate(num_samples, batch_size, device, save_dir=save_dir, **kwargs)
        print(f"Generated {len(structures)} structures.")
        return structures
