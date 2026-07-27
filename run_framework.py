import hydra
from omegaconf import DictConfig
import torch
from sandbox.contracts.base import BaseCrystalModel

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.runner.device)
    model: BaseCrystalModel = hydra.utils.instantiate(cfg.model)
    model.to(device)
    runner = hydra.utils.instantiate(cfg.runner)
    structures = runner(
        model=model,
        num_samples=cfg.runner.num_samples,
        batch_size=cfg.runner.batch_size,
        device=device,
        save_dir=cfg.runner.save_dir
    )
    print(f"Generated {len(structures)} structures.")

if __name__ == "__main__":
    main()
