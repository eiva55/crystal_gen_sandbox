import subprocess
import sys
import os
from pathlib import Path
from sandbox.contracts.base import BaseCrystalModel

class CrystalDiTModel(BaseCrystalModel):
    def __init__(self, ckpt_path="./models/crystaldit/checkpoints/best_model.pt", **kwargs):
        # Преобразуем относительный путь в абсолютный относительно корня проекта
        self.ckpt_path = os.path.abspath(ckpt_path)

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        python_path = os.path.expanduser("~/miniconda3/envs/crystaldit/bin/python")
        cmd = [
            python_path,
            "generate_crystals.py",
            "--checkpoint", self.ckpt_path,
            "--num_samples", str(num_samples),
            "--batch_size", str(batch_size),
            "--output_dir", save_dir or "./outputs/crystaldit",
            "--device", "cpu"
        ]
        process = subprocess.Popen(cmd, cwd="models/crystaldit", stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("CrystalDiT generation failed")
        return [save_dir or "./outputs/crystaldit"]
