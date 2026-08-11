import json
import subprocess
import sys
import os
from sandbox.contracts import BaseCrystalModel
from sandbox.utils.load_structures import load_structure_files


class DiffCSPModel(BaseCrystalModel):
    """Composition/symmetry-conditioned CSP wrapper for DiffCSP++.

    target_composition can be either:
      - a single dict {"atom_types": [...], "spacegroup_number": int,
        "wyckoff_letters": [...]} — replicated num_samples times to get
        several independent samples of the same target;
      - a list of such dicts — used as-is (one structure per entry), for
        composition sweeps where each entry is already a distinct point
        (e.g. varying doping fraction). num_samples is ignored in this case.

    scripts/sample.py requires an ABSOLUTE model_path (hydra
    initialize_config_dir constraint upstream) — resolved in __init__,
    same pattern as ADiTModel/CrystalDiTModel in this repo.
    """

    def __init__(self, ckpt_path="./models/diffcsp_pp/checkpoints/pretrained_models/mp_csp",
                 conda_env="diffcsp_pp", **kwargs):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None,
                 target_composition=None, **kwargs):
        if target_composition is None:
            raise ValueError(
                "DiffCSPModel.generate() requires target_composition — e.g. "
                "{'atom_types': ['Hf','O','O','O','O'], 'spacegroup_number': 33, "
                "'wyckoff_letters': ['4a','4a','4a','4a']}"
            )

        if isinstance(target_composition, dict):
            entries = [target_composition] * num_samples
        else:
            entries = list(target_composition)

        output_dir = os.path.abspath(save_dir or "./outputs/diffcsp")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "targets.json")
        with open(json_path, "w") as f:
            json.dump(entries, f)

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "scripts/sample.py",
            "--model_path", self.ckpt_path,
            "--save_path", output_dir,
            "--json_file", json_path,
        ]
        process = subprocess.Popen(cmd, cwd="models/diffcsp_pp",
                                     stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("DiffCSP++ CSP generation failed")

        return load_structure_files(output_dir, pattern="*.cif")
