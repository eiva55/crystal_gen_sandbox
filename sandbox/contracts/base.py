from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader

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
        """Перемещает модель на устройство (по умолчанию ничего не делает)."""
        return self

class BaseDataset(ABC):
    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Возвращает train, val, test загрузчики."""
        pass

class BaseTask(ABC):
    @abstractmethod
    def run(self, model, dataloaders, **kwargs) -> Dict[str, Any]:
        """Запускает задачу (train/eval/generate)."""
        pass

class BaseMetrics(ABC):
    @abstractmethod
    def compute(self, generated, reference) -> Dict[str, float]:
        """Вычисляет метрики."""
        pass
