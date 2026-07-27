import subprocess
import sys
import os
from datetime import datetime
from sandbox.contracts.base import BaseCrystalModel
from sandbox.utils.load_structures import load_structure_files


class SGEquiDiffModel(BaseCrystalModel):
    def __init__(self, ckpt_dir="./experiments/sgequidiff/mp_20", conda_env="SGEquiDiff", **kwargs):
        self.ckpt_dir = ckpt_dir  # relative to models/sgequidiff (the subprocess's cwd)
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        # Unique per-run tag: SGEquiDiff always writes into ckpt_dir with no
        # separate output-dir flag, so this is the only way to avoid
        # collisions with files from a previous run.
        tag = datetime.now().strftime("%Y%m%d%H%M%S%f")

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "scripts/generate_crystals.py",
            "--num_samples", str(num_samples),
            "--batch_size", str(batch_size),
            "--ckpt_dir", self.ckpt_dir,
            "--load_best_submodules",
            "--temperature", "1.0",
            "--save_method", "poscar",
            "--outfile_name", tag,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.Popen(cmd, cwd="models/sgequidiff", env=env, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("SGEquiDiff generation failed")

        # Resolve ckpt_dir relative to models/sgequidiff — we're back in the
        # project root now, not in the subprocess's cwd.
        search_dir = os.path.join("models/sgequidiff", self.ckpt_dir)
        return load_structure_files(search_dir, pattern=f"POSCAR-*-{tag}")
