import subprocess
import sys
import os
from sandbox.contracts import BaseCrystalModel
from sandbox.utils.load_structures import load_structure_files


class MiADModel(BaseCrystalModel):
    def __init__(self, conda_env="miad", **kwargs):
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        # MiAD has no --seed CLI flag; its own lib/run.py (plain argparse,
        # not Hydra) reads `seed:` from this YAML config instead. We
        # temporarily override it, run, then always restore the original
        # text so a later run without an explicit seed doesn't silently
        # inherit this override.
        config_path = "models/miad/saved_configs/generate_miad_mp20.yaml"
        seed = kwargs.get("seed")
        original_config_text = None

        if seed is not None:
            import yaml
            with open(config_path) as f:
                original_config_text = f.read()
            config_data = yaml.safe_load(original_config_text)
            config_data["seed"] = seed
            with open(config_path, "w") as f:
                yaml.safe_dump(config_data, f)

        try:
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
        finally:
            if original_config_text is not None:
                with open(config_path, "w") as f:
                    f.write(original_config_text)

        return load_structure_files("models/miad/saved_results/generate_miad_mp20/generated_crystals_cif", pattern="*.cif")
