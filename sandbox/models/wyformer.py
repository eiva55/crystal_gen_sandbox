import subprocess
import sys
from sandbox.contracts.base import BaseCrystalModel


class WyFormerModel(BaseCrystalModel):
    def __init__(self, hf_model="SymmetryAdvantage/WyFormer-Alex-MP20", conda_env="WyFormer", **kwargs):
        self.hf_model = hf_model
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        cmd = [
            "wyformer-generate",
            "generated_structures.json.gz",
            "--hf-model", self.hf_model,
            "--device", "cpu",
            "--firm-n-samples", str(num_samples)
        ]
        full_cmd = f"conda run -n {self.conda_env} " + " ".join(cmd)
        process = subprocess.Popen(full_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("WyFormer generation failed")
        return ["generated_structures.json.gz"]
