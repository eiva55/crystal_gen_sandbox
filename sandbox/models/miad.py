import subprocess
import sys
import os
from sandbox.contracts.base import BaseCrystalModel
from sandbox.utils.load_structures import load_structure_files


class MiADModel(BaseCrystalModel):
    def __init__(self, conda_env="miad", **kwargs):
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "lib/run.py",
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
        return load_structure_files("models/miad/saved_results/generate_miad_mp20/generated_crystals_cif", pattern="*.cif")
