import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader

from sandbox.utils.check import CheckResult


class BaseCrystalModel(ABC):
    @abstractmethod
    def generate(self, num_samples: int, batch_size: int, device: torch.device, **kwargs) -> List[Any]:
        """Генерирует кристаллы."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        pass

    def to(self, device):
        return self

    def sanity_check(self) -> List[CheckResult]:
        """Verify wiring (conda env, checkpoint/data paths) without running
        real generation. Generic by design: it inspects instance attributes
        rather than requiring each of the five model subclasses to implement
        its own check — a new model gets this for free just by setting
        `self.conda_env` / `*_path` / `*_dir` attributes in __init__.
        """
        results = []

        conda_env = getattr(self, "conda_env", None)
        if conda_env:
            try:
                proc = subprocess.run(
                    ["conda", "run", "-n", conda_env, "python", "-c", "print('ok')"],
                    capture_output=True, text=True, timeout=30,
                )
                ok = proc.returncode == 0
                message = "passed" if ok else proc.stderr.strip()[:300]
            except Exception as exc:
                ok, message = False, str(exc)
            results.append(CheckResult(name=f"conda_env:{conda_env}", ok=ok, message=message))

        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, str) and any(k in attr_name.lower() for k in ("path", "dir", "ckpt")):
                exists = os.path.exists(attr_value)
                results.append(CheckResult(
                    name=f"path:{attr_name}",
                    ok=exists,
                    message=attr_value if exists else f"missing: {attr_value}",
                ))

        return results


class BaseDataset(ABC):
    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass


class BaseTask(ABC):
    @abstractmethod
    def run(self, model, dataloaders, **kwargs) -> Dict[str, Any]:
        pass


class BaseMetrics(ABC):
    @abstractmethod
    def compute(self, generated, reference) -> Dict[str, float]:
        pass
