"""Trivial baseline sharing BaseCrystalModel's contract.

Used to sanity-check the metrics pipeline: a baseline that copies existing
MP-20 structures verbatim must score near-zero novelty. If it doesn't, the
novelty metric itself is broken, not the real generative models.
"""
import random
from sandbox.contracts import BaseCrystalModel
from sandbox.datasets.mp20 import MP20Dataset


class RandomMP20CopyBaseline(BaseCrystalModel):
    def __init__(self, mp20_root="./models/adit/data/mp_20", **kwargs):
        self.mp20_root = mp20_root

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        dataset = MP20Dataset(root=self.mp20_root, split="test", limit=max(num_samples * 5, 50))
        sampled = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        return [dataset[i] for i in sampled]
