import json
import os
import random

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.tensorboard import SummaryWriter

from sandbox.contracts import BaseCrystalModel


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible generation runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    seed = int(cfg.get("random_seed", {}).get("seed", 42))
    set_global_seed(seed)

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
    metrics = hydra.utils.instantiate(cfg.metrics) if "metrics" in cfg else None
    viz_enabled = cfg.get("viz", {}).get("enabled", False)

    save_dir = cfg.runner.get("save_dir")
    model_name = cfg.model.get("_target_", "unknown")

    logger = SummaryWriter(log_dir=os.path.join(save_dir, "tb")) if save_dir else None

    try:
        result = hydra.utils.instantiate(
            cfg.runner, task=task, model=model, device=device, dataset=dataset,
            metrics=metrics, viz_enabled=viz_enabled, seed=seed
        )

        if logger is not None:
            logger.add_text("run/model", model_name)
            logger.add_text("run/seed", str(seed))
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        logger.add_scalar(f"metrics/{key}", value, global_step=seed)
            elif isinstance(result, list):
                logger.add_scalar("generation/num_structures", len(result), global_step=seed)

        if save_dir and isinstance(result, dict):
            os.makedirs(save_dir, exist_ok=True)
            metrics_path = os.path.join(save_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved metrics to {metrics_path}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            summary_path = os.path.join(save_dir, "run_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"model: {model_name}\n")
                f.write(f"seed: {seed}\n")
                if isinstance(result, dict):
                    f.write(f"metrics: {result}\n")
                elif isinstance(result, list):
                    f.write(f"num_structures_generated: {len(result)}\n")
            print(f"Saved run summary to {summary_path}")
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
