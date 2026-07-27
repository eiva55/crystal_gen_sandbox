import subprocess
import sys
import os
from pathlib import Path
from sandbox.contracts.base import BaseCrystalModel

class WyFormerModel(BaseCrystalModel):
    def __init__(self, hf_model="SymmetryAdvantage/WyFormer-Alex-MP20", **kwargs):
        self.hf_model = hf_model
        self._loaded = False

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        # Используем оригинальный CLI
        cmd = [
            "wyformer-generate",
            "generated_structures.json.gz",
            "--hf-model", self.hf_model,
            "--device", "cpu",
            "--firm-n-samples", str(num_samples)
        ]
        full_cmd = "conda run -n WyFormer " + " ".join(cmd)
        process = subprocess.Popen(full_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("WyFormer generation failed")
        return ["generated_structures.json.gz"]  # возвращаем путь
