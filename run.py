import hydra
from omegaconf import DictConfig
import torch

from sandbox.contracts.base import BaseCrystalModel


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.runner.device)

    task = hydra.utils.instantiate(cfg.task)
    model: BaseCrystalModel = hydra.utils.instantiate(cfg.model)
    model.to(device)

    hydra.utils.instantiate(cfg.runner, task=task, model=model, device=device)


if __name__ == "__main__":
    main()
