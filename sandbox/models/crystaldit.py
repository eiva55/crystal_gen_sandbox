import subprocess
import sys
import os
from sandbox.contracts.base import BaseCrystalModel


class CrystalDiTModel(BaseCrystalModel):
    def __init__(self, ckpt_path="./models/crystaldit/checkpoints/best_model.pt", conda_env="crystaldit", **kwargs):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "generate_crystals.py",
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
