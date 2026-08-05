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
        config_path = "models/miad/saved_configs/generate_miad_mp20.yaml"
        lock_path = config_path + ".lock"
        seed = kwargs.get("seed")

        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"{lock_path} already exists — another MiADModel.generate() "
                f"call appears to be in progress (or a previous one crashed "
                f"without cleaning up). Wait for it to finish, or if you're "
                f"certain nothing is running, delete the lock file manually "
                f"and verify {config_path} still has its original values "
                f"before retrying."
            )
        os.close(lock_fd)

        try:
            import yaml
            with open(config_path) as f:
                original_config_text = f.read()
            config_data = yaml.safe_load(original_config_text)
            if seed is not None:
                config_data["seed"] = seed
            config_data["num_samples"] = num_samples
            if batch_size is not None:
                config_data["data"]["batch_size"] = batch_size
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
                with open(config_path, "w") as f:
                    f.write(original_config_text)
        finally:
            os.remove(lock_path)

        return load_structure_files("models/miad/saved_results/generate_miad_mp20/generated_crystals_cif", pattern="*.cif")
