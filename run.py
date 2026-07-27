import hydra
from omegaconf import DictConfig
import torch

from sandbox.contracts import BaseCrystalModel


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.runner.device)

    task = hydra.utils.instantiate(cfg.task)
    model: BaseCrystalModel = hydra.utils.instantiate(cfg.model)
    model.to(device)

    if cfg.get("sanity_run", False):
        results = model.sanity_check()
        if not results:
            print("No sanity checks applicable for this model (no conda_env/path attributes found).")
            return
        for r in results:
            status = "OK" if r.ok else "FAIL"
            print(f"[{status}] {r.name}: {r.message}")
        if not all(r.ok for r in results):
            raise SystemExit(1)
        return

    dataset = hydra.utils.instantiate(cfg.dataset) if "dataset" in cfg else None
    viz_enabled = cfg.get("viz", {}).get("enabled", False)

    hydra.utils.instantiate(
        cfg.runner, task=task, model=model, device=device, dataset=dataset, viz_enabled=viz_enabled
    )


if __name__ == "__main__":
    main()
