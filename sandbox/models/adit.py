import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch
from sandbox.contracts.base import BaseCrystalModel

ADIT_ROOT = Path(__file__).parent.parent.parent / "models" / "adit"
sys.path.insert(0, str(ADIT_ROOT))

from src.models.ldm_module import LatentDiffusionLitModule

class ADiTModel(BaseCrystalModel):
    def __init__(
        self,
        ckpt_path: str = None,
        autoencoder_ckpt: str = None,
        data: str = "mp20_only",
        sampling: dict = None,
        **kwargs
    ):
        super().__init__()
        default_sampling = {"num_samples": 10, "batch_size": 10, "cfg_scale": 2.0, "visualize": True}
        if sampling:
            default_sampling.update(sampling)
        
        cfg_dict = {
            "ckpt_path": ckpt_path,
            "autoencoder_ckpt": autoencoder_ckpt,
            "data": data,
            "sampling": default_sampling,
            "denoiser": {
                "_target_": "src.models.denoisers.dit.DiT",
                "d_x": 8,
                "d_model": 768,
                "nhead": 12,
                "num_layers": 12,
                "num_datasets": 2,
            },
            "interpolant": {
                "_target_": "src.models.interpolants.flow_matching.FlowMatchingInterpolant",
                "min_t": 0.01,
                "corrupt": True,
                "num_timesteps": 100,
                "self_condition": True,
                "self_condition_prob": 0.5,
            },
            "augmentations": {"frac_coords": True, "pos": True},
            "conditioning": {"dataset_idx": True, "spacegroup": False},
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "_partial_": True,
                "lr": 0.0001,
                "weight_decay": 0.0,
            },
            "scheduler": None,
            "scheduler_frequency": 250,
            "compile": False,
        }
        self.cfg = OmegaConf.create(cfg_dict)
        self.model = LatentDiffusionLitModule(self.cfg)
        if ckpt_path:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)

    def generate(self, num_samples, batch_size, device, save_dir=None, **kwargs):
        if num_samples:
            self.cfg.sampling.num_samples = num_samples
        if batch_size:
            self.cfg.sampling.batch_size = batch_size
        return self.model.generate(
            num_samples=self.cfg.sampling.num_samples,
            batch_size=self.cfg.sampling.batch_size,
            device=device
        )

    def to(self, device):
        self.model.to(device)
        return self
