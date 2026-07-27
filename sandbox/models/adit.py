import subprocess
import sys
import os
from sandbox.contracts.base import BaseCrystalModel


class ADiTModel(BaseCrystalModel):
    def __init__(self, ckpt_path="./models/adit/checkpoints/adit/ldm.ckpt",
                 autoencoder_ckpt="./models/adit/checkpoints/adit/vae.ckpt",
                 data="mp20_only", conda_env="ADiT", **kwargs):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.autoencoder_ckpt = os.path.abspath(autoencoder_ckpt)
        self.data = data
        self.conda_env = conda_env

    def load_checkpoint(self, path: str):
        pass

    def save_checkpoint(self, path: str):
        pass

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "src/eval_diffusion.py",
            f"ckpt_path={self.ckpt_path}",
            f"diffusion_module.autoencoder_ckpt={self.autoencoder_ckpt}",
            f"data={self.data}",
            "trainer.accelerator=cpu",
            "trainer.devices=1",
            f"diffusion_module.sampling.num_samples={num_samples}",
            f"diffusion_module.sampling.batch_size={batch_size}"
        ]
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        process = subprocess.Popen(cmd, cwd="models/adit", env=env, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("ADiT generation failed")
        return [save_dir or "./outputs/adit_test"]
