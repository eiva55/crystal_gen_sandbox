from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch

class BaseCrystalModel(ABC):
    @abstractmethod
    def generate(self, num_samples: int, batch_size: int, device: torch.device, **kwargs) -> List[Any]:
        """Генерирует кристаллы."""
        pass

class BaseDataset(ABC):
    @abstractmethod
    def load_data(self):
        """Загружает и предобрабатывает данные."""
        pass

class BaseTask(ABC):
    @abstractmethod
    def run(self, model, dataset, **kwargs):
        """Запускает задачу (например, генерацию)."""
        pass

class BaseMetrics(ABC):
    @abstractmethod
    def compute(self, generated, reference):
        """Вычисляет метрики."""
        pass
