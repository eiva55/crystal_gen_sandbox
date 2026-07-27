import logging
from pathlib import Path
import yaml
import os

import hydra
from omegaconf import OmegaConf
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import wandb

from aliases import *
from custom_dataset import AsymmetricUnitDataset
from utils.io_utils import setup_experiment_directory, get_now_str
from utils.logging_utils import setup_logger
from utils.train_utils import (
    model_size,
    get_num_trainable_parameters,
    experiment_setup,
    dispatch_model,
)
from model.crystal_sampler import CrystalSampler
from trainer import TrainerConfig, Trainer


def ddp_setup():
    init_process_group(backend="nccl")


def main(config: DictConfig, experiment_directory: Path):
    # asu dictionary, embeddings, and seeding
    experiment_setup(config)

    # artifact tracking
    log_level: int = logging.DEBUG if config.diagnostics.debug else logging.INFO
    logger = setup_logger(
        __name__, level=log_level, custom_handle=experiment_directory / "log.out"
    )
    config.diagnostics.experiment_directory = experiment_directory

    # optionally, configure the process group
    if config.compute.distributed:
        local_rank: int = int(os.environ["LOCAL_RANK"])
        logger.info(f"{local_rank=} online")
        ddp_setup()
        logger.info("configure DDP")
        device: torch.device = torch.device(local_rank)
    else:
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    distributed_guard: bool = (not config.compute.distributed) or (local_rank == 0)

    # initialize wandb
    if distributed_guard:
        config_path = experiment_directory.as_posix() + "/config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(dict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True)), f)

        if config.wandb.use_wandb:
            wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            project_name: str = (
                f"{config.wandb.run_name_prefix}_generative_crystal_{get_now_str()}"
            )
            wandb_kwargs = {}
            if config.wandb.resume_from_id is not None:
                wandb_kwargs["resume"] = "allow"
                wandb_kwargs["id"] = config.wandb.resume_from_id
            wandb.init(
                config=wandb.config,
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=project_name,
                **wandb_kwargs,
            )

            # Push up the config as a yaml file
            artifact = wandb.Artifact(name="config_artifact", type="config")
            artifact.add_file(config_path)
            wandb.log_artifact(artifact)
    logger.info(f"{experiment_directory.as_posix()=}")
    logger.info(f"{OmegaConf.to_yaml(config)}")

    # load dataset
    logger.info("Loading dataset")
    train_dataset = AsymmetricUnitDataset(
        name=config.dataset.name,
        split="train",
        device=device,
    )
    val_dataset = AsymmetricUnitDataset(
        name=config.dataset.name,
        split="val",
        device=device,
    )
    if config.experiment.n_crystals_to_overfit_to > 0:
        # Overfit to a few crystals for sanity checking
        assert config.optim.batch_size <= config.experiment.n_crystals_to_overfit_to
        # train_dataset.reindex(torch.randperm(len(train_dataset)))
        train_dataset.reindex(torch.arange(config.experiment.n_crystals_to_overfit_to))
        val_dataset.reindex([0])
    logger.info(f"{len(train_dataset)} training crystals.")
    logger.info(f"{len(val_dataset)} validation crystals.")
    logger.info("Loaded dataset")

    train_sampler = (
        DistributedSampler(train_dataset) if config.compute.distributed else None
    )
    val_sampler = DistributedSampler(val_dataset) if config.compute.distributed else None
    shuffle: bool = False if config.compute.distributed else True
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=config.optim.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=lambda x: x,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=config.optim.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=lambda x: x,
    )

    # instantiate the model
    model: CrystalSampler = dispatch_model(config, device)
    model.space_group_sampler.reinit_and_freeze_probs(
        train_dataset.empirical_space_group_probs
    )
    size: str = model_size(model, as_str=True)
    logger.info(f"instantiated model with {size=} and {get_num_trainable_parameters(model)} trainable parameters.")

    # train
    trainer_config: TrainerConfig = TrainerConfig(
        model=model,
        device=device,
        hydra_config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        logger=logger,
    )
    trainer: Trainer = Trainer(trainer_config)
    trainer.fit_offline(num_epochs=config.optim.num_epochs)

    if config.compute.distributed:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path="../configurations",
    config_name="default",
)
def wrapped_main(config: DictConfig) -> None:
    """
    Wraps the nominal `main` function to (1) use the Hydra CLI API and
    (2) construct the config and I/O artifacts outside of `main` for use
    with distributed training.
    """
    if config.diagnostics.experiment_directory in [None, "None"]:
        shared_experiment_directory: Path = Path(
            setup_experiment_directory(
                application="generative_crystals",
                leaf_directory_modifier=config.wandb.run_name_prefix,
            )
        )
    else:
        shared_experiment_directory: Path = Path(config.diagnostics.experiment_directory)
    main(config, shared_experiment_directory)


if __name__ == "__main__":
    wrapped_main()
