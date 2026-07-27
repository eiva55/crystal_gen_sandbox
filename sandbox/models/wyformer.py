import os
import subprocess
import sys
from pathlib import Path

from sandbox.contracts.base import BaseCrystalModel
from sandbox.utils.load_structures import load_structure_files


class WyFormerModel(BaseCrystalModel):
    def __init__(self, hf_model="SymmetryAdvantage/WyFormer-Alex-MP20", conda_env="WyFormer", **kwargs):
        self.hf_model = hf_model
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def extra_checks(self):
        from sandbox.utils.check import CheckResult
        import subprocess
        try:
            proc = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "python", "-c", "import pyxtal"],
                capture_output=True, text=True, timeout=30,
            )
            ok = proc.returncode == 0
            message = "passed" if ok else proc.stderr.strip()[:300]
        except Exception as exc:
            ok, message = False, str(exc)
        return [CheckResult(name="pyxtal_importable", ok=ok, message=message)]

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        raw_output = "generated_structures.json.gz"
        cmd = [
            "wyformer-generate",
            raw_output,
            "--hf-model", self.hf_model,
            "--device", "cpu",
            "--firm-n-samples", str(num_samples),
        ]
        full_cmd = f"conda run -n {self.conda_env} " + " ".join(cmd)
        process = subprocess.Popen(full_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("WyFormer generation failed")

        raw_path = os.path.join("models/wyformer", raw_output)
        recon_dir = os.path.abspath(save_dir or "./outputs/wyformer")
        reconstruct_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "wyformer_reconstruct.py")
        )
        recon_cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", reconstruct_script,
            "--input", os.path.abspath(raw_path),
            "--output_dir", recon_dir,
        ]
        recon_process = subprocess.Popen(recon_cmd, stdout=sys.stdout, stderr=sys.stderr)
        recon_process.wait()
        if recon_process.returncode != 0:
            raise RuntimeError("WyFormer Wyckoff-to-structure reconstruction failed")

        return load_structure_files(recon_dir, pattern="*.cif")
