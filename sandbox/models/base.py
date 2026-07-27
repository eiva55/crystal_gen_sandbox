from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional, Tuple, List

class BaseCrystalModel(ABC):
    """Абстрактный базовый класс для всех генеративных моделей кристаллов."""

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Прямой проход для обучения."""
        pass

    @abstractmethod
    def generate(
        self,
        num_samples: int,
        batch_size: int,
        device: torch.device,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> List[Any]:
        """Генерация новых кристаллов."""
        pass

    @abstractmethod
    def configure_optimizers(self, cfg: Dict[str, Any]) -> Tuple[Optional[torch.optim.Optimizer], Optional[Any]]:
        """Настройка оптимизатора (если нужно)."""
        pass

    def save_checkpoint(self, path: str) -> None:
        """Сохраняет состояние модели."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Загружает состояние модели."""
        self.load_state_dict(torch.load(path, map_location='cpu'))
