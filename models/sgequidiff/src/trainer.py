import dataclasses
from logging import Logger

import wandb
import torch
from torch.utils.data import DataLoader
from torch.distributed import barrier
from torch.nn.parallel import DistributedDataParallel as DDP

from aliases import *
from crystal_classes import ASUCrystal, CrystalDict
from model.crystal_sampler import CrystalSampler
from utils.train_utils import (
    configure_optimizer,
    configure_scheduler,
    AverageMeter,
    log_model_weight_and_grad_norms,
    diagnose_and_zero_out_nan_gradients,
)


@dataclasses.dataclass
class TrainerConfig:
    model: CrystalSampler
    device: torch.device
    hydra_config: DictConfig

    train_dataloader: DataLoader = None
    val_dataloader: DataLoader = None

    logger: Logger = None


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.hydra_config = self.config.hydra_config
        self.distributed: bool = self.hydra_config.compute.distributed

        self.model: CrystalSampler = self.config.model
        self.train_dataloader: DataLoader = self.config.train_dataloader
        self.val_dataloader: DataLoader = self.config.val_dataloader

        # configure optimizer
        self.optimizer: Optimizer = configure_optimizer(
            self.hydra_config, self.model
        )
        self.scheduler: LRScheduler = configure_scheduler(
            self.hydra_config, self.optimizer
        )

        # fault tolerance: restart from snapshots if available
        self.epochs_completed: int = 0
        snapshots: List[Path] = list(
            Path(self.hydra_config.diagnostics.experiment_directory).glob("**/*snapshot-*")
        )
        if len(snapshots) > 0:
            snapshot_versions: List[int] = [
                int(snapshot.with_suffix("").as_posix().split("-")[-1])
                for snapshot in snapshots
            ]
            most_recent_snapshot: Path = snapshots[
                max(enumerate(snapshot_versions), key=lambda x: x[1])[0]
            ]
            self._load_snapshot(most_recent_snapshot)  # load model and optimizer

        # optionally, configure DDP
        if self.distributed:
            self.local_rank: int = int(os.environ["LOCAL_RANK"])
            self.global_rank: int = int(os.environ["RANK"])
            self.device: torch.device = torch.device(self.local_rank)
            self.model.to(self.local_rank)
            self.model: Module = DDP(self.model, device_ids=[self.local_rank])
            self.model_module: CrystalSampler = self.model.module
        else:
            self.device: torch.device = self.config.device
            self.model.to(self.device)
            self.model_module: CrystalSampler = self.model

        self.distributed_guard: bool = (not self.distributed) or (self.local_rank == 0)

        # diagnostics
        self.checkpoint_every: int = self.hydra_config.diagnostics.checkpoint_every
        self.logger: callable = self.config.logger.info if (self.config.logger is not None) else print
        self.meters: Dict = {
            "train loss": AverageMeter(),
            "train space group log prob": AverageMeter(),
            "train lattice log prob": AverageMeter(),
            "train wyckoffs log prob": AverageMeter(),
            "train elements log prob": AverageMeter(),
            "train termination log prob": AverageMeter(),
            "train score matching loss": AverageMeter(),
            "val loss": AverageMeter(),
            "val space group log prob": AverageMeter(),
            "val lattice log prob": AverageMeter(),
            "val wyckoffs log prob": AverageMeter(),
            "val elements log prob": AverageMeter(),
            "val termination log prob": AverageMeter(),
            "val score matching loss": AverageMeter(),
        }
        self.best_metrics: Dict = {
            "val space group log prob": -1e8,
            "val lattice log prob": -1e8,
            "val wyckoff transformer log probs": -1e8,
            "val score matching loss": 1e8,
        }

    def _load_snapshot(self, snapshot_path: Path) -> None:
        """Instantiates `self.model` with parameters saved at a previous checkpoint.

        Parameters
        ----------
        snapshot_path : Path
            path to the snapshot `.pt` file
        """
        # TODO: optionally load submodules individually. See scripts/generate_crystals.py
        snapshot: dict = torch.load(snapshot_path, map_location="cpu")
        self.logger(f"Loaded snapshot from: {snapshot_path.absolute().as_posix()}")

        self.epochs_completed: int = snapshot["step"]
        self.logger(f"Snapshot epochs completed: {self.epochs_completed}")

        self.model.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])

    def _save_snapshot(self, step: int, save_best_submodules: bool = False):
        """Saves the current model state to a `.pt` binary within the experiment
        directory specified in `self.hydra_config.diagnostics.experiment_directory`.

        Parameters
        ----------
        step : int
            non-negative integer specifying which step we've completed in the optimization
            procdure (used to infer the most recent snapshot to load from).
        """
        if (
            not save_best_submodules
            and step != 0
            and step % self.hydra_config.diagnostics.checkpoint_every == 0
            and self.distributed_guard
        ):
            milestone: int = step // self.hydra_config.diagnostics.checkpoint_every
            snapshot: dict = {
                "step": step,
                "model": self.model_module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            snapshot_path: Path = (
                self.hydra_config.diagnostics.experiment_directory / f"snapshot-{milestone}.pt"
            )
            torch.save(snapshot, snapshot_path)
            self.logger(f"Saved snapshot to: {snapshot_path.as_posix()}")
            if self.hydra_config.wandb.use_wandb:
                wandb.save(snapshot_path.as_posix())
        elif save_best_submodules:
            if (
                self.meters["val space group log prob"].count > 0 and
                self.meters["val space group log prob"].avg
                > self.best_metrics["val space group log prob"]
            ):
                self.best_metrics["val space group log prob"]: float = (
                    self.meters["val space group log prob"].avg
                )
                snapshot = self.model_module.space_group_sampler.state_dict()
                snapshot_path: Path = (
                    self.hydra_config.diagnostics.experiment_directory
                    / f"best_space_group_snapshot.pt"
                )
                torch.save(snapshot, snapshot_path)
                self.logger(f"Saved space group snapshot to: {snapshot_path.as_posix()}")
                if self.hydra_config.wandb.use_wandb:
                    wandb.save(snapshot_path.as_posix())
            if (
                self.meters["val lattice log prob"].count > 0 and
                self.meters["val lattice log prob"].avg
                > self.best_metrics["val lattice log prob"]
            ):
                self.best_metrics["val lattice log prob"]: float = (
                    self.meters["val lattice log prob"].avg
                )
                snapshot = self.model_module.lattice_sampler.state_dict()
                snapshot_path: Path = (
                    self.hydra_config.diagnostics.experiment_directory
                    / f"best_lattice_snapshot.pt"
                )
                torch.save(snapshot, snapshot_path)
                self.logger(f"Saved lattice snapshot to: {snapshot_path.as_posix()}")
                if self.hydra_config.wandb.use_wandb:
                    wandb.save(snapshot_path.as_posix())

            wyckoff_transformer_aggregate_log_probs = (
                self.meters["val wyckoffs log prob"].avg
                + self.meters["val elements log prob"].avg
                + self.meters["val termination log prob"].avg
            )
            if (
                self.meters["val wyckoffs log prob"].count > 0 and
                wyckoff_transformer_aggregate_log_probs
                > self.best_metrics["val wyckoff transformer log probs"]
            ):
                self.best_metrics["val wyckoff transformer log probs"]: float = (
                    wyckoff_transformer_aggregate_log_probs
                )
                snapshot = self.model_module.wyckoff_and_element_sampler.state_dict()
                snapshot_path: Path = (
                    self.hydra_config.diagnostics.experiment_directory
                    / f"best_wyckoff-transformer_snapshot.pt"
                )
                torch.save(snapshot, snapshot_path)
                self.logger(f"Saved wyckoff-transformer snapshot to: {snapshot_path.as_posix()}")
                if self.hydra_config.wandb.use_wandb:
                    wandb.save(snapshot_path.as_posix())
            if (
                self.meters["val score matching loss"].count > 0 and
                self.meters["val score matching loss"].avg
                < self.best_metrics["val score matching loss"]
            ):
                self.best_metrics["val score matching loss"]: float = (
                    self.meters["val score matching loss"].avg
                )
                snapshot = self.model_module.atom_coord_diffusion_model.state_dict()
                snapshot_path: Path = (
                    self.hydra_config.diagnostics.experiment_directory
                    / f"best_diffusion_snapshot.pt"
                )
                torch.save(snapshot, snapshot_path)
                self.logger(f"Saved diffusion snapshot to: {snapshot_path.as_posix()}")
                if self.hydra_config.wandb.use_wandb:
                    wandb.save(snapshot_path.as_posix())

    @torch.no_grad()
    def update_meters(self, loss: Tensor, artifacts: Dict, train: bool):
        prefix = "train" if train else "val"
        self.meters[f"{prefix} loss"].update(loss)
        self.meters[f"{prefix} space group log prob"].update(
            artifacts["space_group_log_prob"].mean()
        )
        self.meters[f"{prefix} lattice log prob"].update(
            artifacts["lattice_log_prob"].mean()
        )
        self.meters[f"{prefix} wyckoffs log prob"].update(
            artifacts["wyckoffs_log_prob"].mean()
        )
        self.meters[f"{prefix} elements log prob"].update(
            artifacts["elements_log_prob"].mean()
        )
        self.meters[f"{prefix} termination log prob"].update(
            artifacts["termination_log_prob"].mean()
        )
        self.meters[f"{prefix} score matching loss"].update(
            artifacts["score_matching_loss"].mean()
        )

    @torch.no_grad()
    def log_metrics(self, loss: Tensor, batch_number: int, artifacts: Dict):
        metric_str: str = f"Iteration [{batch_number:04d}] --- Loss: {loss:0.4f}"
        if self.distributed:
            self.logger(f"GPU {self.local_rank=} " + metric_str)
        else:
            self.logger(metric_str)

        if self.hydra_config.wandb.use_wandb and self.distributed_guard:
            wandb.log(
                {
                    "train loss": loss,
                    "train space group log prob": artifacts["space_group_log_prob"].mean(),
                    "train lattice log prob": artifacts["lattice_log_prob"].mean(),
                    "train elements log prob": artifacts["elements_log_prob"].mean(),
                    "train wyckoffs log prob": artifacts["wyckoffs_log_prob"].mean(),
                    "train termination log prob": artifacts["termination_log_prob"].mean(),
                    "train score matching loss": artifacts["score_matching_loss"].mean(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }, step=batch_number)
            if self.distributed_guard:
                log_model_weight_and_grad_norms(batch_number, self.model_module)

    def run_batch(
        self,
        crystals: CrystalDict,
        batch_number: int = None,
        train: bool = True
    ) -> Tensor:
        loss, artifacts = self.model_module.compute_loss(crystals)

        _loss = loss.detach()
        self.update_meters(_loss, artifacts, train)
        if train:
            loss.backward()
            if batch_number % self.hydra_config.diagnostics.log_freq == 0:
                self.log_metrics(_loss, batch_number, artifacts)
        return loss

    def fit_offline(self, num_epochs: int):
        if (
            self.hydra_config.wandb.use_wandb
            and self.hydra_config.wandb.watch_grads
            and self.distributed_guard
        ):
            wandb.watch(
                self.model_module,
                log="all",
                log_freq=self.hydra_config.diagnostics.log_freq,
            )

        batch_number = self.epochs_completed * len(self.train_dataloader)
        # num_epochs_without_improvement = 0  # for early stopping
        for epoch_number in range(self.epochs_completed, num_epochs):
            # ------ Train Epoch Loop -------
            self.model_module.train()
            for crystals in self.train_dataloader:
                crystals: CrystalDict = crystals.to_(self.device)
                loss = self.run_batch(crystals, batch_number=batch_number, train=True)

                if self.distributed:
                    barrier()
                    diagnose_and_zero_out_nan_gradients(
                        self.model_module, self.logger, batch_number
                    )

                if self.hydra_config.optim.gradient_clip_algorithm == "value":
                    torch.nn.utils.clip_grad_value_(
                        self.model_module.parameters(),
                        self.hydra_config.optim.gradient_clip_val
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_number += 1

            self.scheduler.step(self.meters["train loss"].avg)
            self.epochs_completed += 1
            self._save_snapshot(epoch_number)
            # ------ End Train Epoch Loop ------

            # ----- Validation Loop -----
            self.model_module.eval()
            if epoch_number % self.hydra_config.diagnostics.validate_every != 0:
                continue
            with torch.no_grad():
                best_val_loss = 1e8
                for crystals in self.val_dataloader:
                    crystals: CrystalDict = crystals.to_(self.device)
                    self.run_batch(crystals, train=False)

                self._save_snapshot(epoch_number, save_best_submodules=True)
                if self.hydra_config.wandb.use_wandb and self.distributed_guard:
                    wandb.log(
                        {
                            "val loss": self.meters["val loss"].avg,
                            "val space group log prob": self.meters["val space group log prob"].avg,
                            "val lattice log prob": self.meters["val lattice log prob"].avg,
                            "val wyckoffs log prob": self.meters["val wyckoffs log prob"].avg,
                            "val elements log prob": self.meters["val elements log prob"].avg,
                            "val termination log prob": self.meters["val termination log prob"].avg,
                            "val score matching loss": self.meters["val score matching loss"].avg,
                        }, step=batch_number
                    )

            for meter in self.meters.values():
                meter.reset()
