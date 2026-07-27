import subprocess
import sys
import os
from pathlib import Path
from sandbox.contracts.base import BaseCrystalModel

class MiADModel(BaseCrystalModel):
    def __init__(self, **kwargs):
        pass

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        # Используем прямой вызов интерпретатора из окружения miad
        python_path = os.path.expanduser("~/miniconda3/envs/miad/bin/python")
        cmd = [
            python_path,
            "lib/run.py",
            "-gpu", "0",
            "-ignore_warnings", "1",
            "-config", "generate_miad_mp20.yaml"
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.Popen(cmd, cwd="models/miad", env=env, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("MiAD generation failed")
        return ["saved_results/generate_miad_mp20/generated_crystals_cif/"]
