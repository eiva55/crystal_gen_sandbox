import subprocess
import sys
import os
from pathlib import Path
from sandbox.contracts.base import BaseCrystalModel

class SGEquiDiffModel(BaseCrystalModel):
    def __init__(self, ckpt_dir=None, **kwargs):
        # Если путь не указан, используем стандартный
        if ckpt_dir is None:
            ckpt_dir = "./experiments/sgequidiff/mp_20"
        # Преобразуем в абсолютный путь
        self.ckpt_dir = os.path.abspath(ckpt_dir)

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        # Используем conda run
        cmd = [
            "conda", "run", "-n", "SGEquiDiff",
            "python", "scripts/generate_crystals.py",
            "--num_samples", str(num_samples),
            "--batch_size", str(batch_size),
            "--ckpt_dir", self.ckpt_dir,
            "--load_best_submodules",
            "--temperature", "1.0"
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.Popen(cmd, cwd="models/sgequidiff", env=env, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("SGEquiDiff generation failed")
        return [self.ckpt_dir]
