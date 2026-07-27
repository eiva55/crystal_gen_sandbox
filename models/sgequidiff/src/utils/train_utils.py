import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from aliases import DictConfig
from crystal_classes import ImmutableASUCrystal, ASUCrystal
from constants import *
from aliases import *
from utils.embedding_utils import set_global_embedding_tools
from utils.io_utils import human_bytes_str
from utils.data_utils import frac_to_cart_coords
from model.crystal_sampler import (
    CrystalSamplerConfig,
    CrystalSampler,
)
from model.lattice_sampler import LatticeSamplerConfig
from model.wyckoff_and_element_transformer import WyckoffElementTransformerConfig
from model.diffusion.diffusion_model import EquivariantDiffusionModelConfig


def get_crystal_sampler_config(config: DictConfig) -> CrystalSamplerConfig:
    lattice_config = LatticeSamplerConfig(**config.model.lattice_config)
    wyckoff_element_config = WyckoffElementTransformerConfig(**config.model.wyckoff_element_config)
    frac_coord_config = EquivariantDiffusionModelConfig(**config.model.frac_coord_config)
    model_config = CrystalSamplerConfig(
        diffusion_model_config=frac_coord_config,
        lattice_model_config=lattice_config,
        transformer_config=wyckoff_element_config,
        lattice_length_noise=config.model.lattice_length_noise,
        lattice_angle_noise=config.model.lattice_angle_noise,
        space_group_grad_weight=config.model.space_group_grad_weight,
        lattice_grad_weight=config.model.lattice_grad_weight,
        wyckoff_element_grad_weight=config.model.wyckoff_element_grad_weight,
        frac_coord_grad_weight=config.model.frac_coord_grad_weight,
    )
    return model_config


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizer(config: DictConfig, model: CrystalSampler) -> Optimizer:
    """Dynamic dispatch helper to configure the optimizer given a configuration and a model.

    Parameters
    ----------
    config : DictConfig
        Hydra configuration
    model : torch.nn.Module
        model whose parameters whould be tracked with the optimizer

    Returns
    -------
    torch.optim.Optimizer
        configured optimizer
    """
    param_groups = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "space_group_sampler" in n:
                param_groups.append(
                    {"params": p, "lr": config.optim.optimizer.space_group_lr}
                )
            elif "lattice_sampler" in n:
                param_groups.append(
                    {"params": p, "lr": config.optim.optimizer.lattice_lr}
                )
            elif "wyckoff_and_element_sampler" in n:
                param_groups.append(
                    {"params": p, "lr": config.optim.optimizer.wyckoff_transformer_lr}
                )
            elif "atom_coord_diffusion_model" in n:
                param_groups.append(
                    {"params": p, "lr": config.optim.optimizer.diffusion_lr}
                )
            else:
                param_groups.append(
                    {"params": p, "lr": config.optim.optimizer.default_lr}
                )

    if config.optim.optimizer.name == "AdamW":
        optimizer: Optimizer = optim.AdamW(
            param_groups,
            lr=config.optim.optimizer.default_lr,
            weight_decay=config.optim.optimizer.weight_decay,
            eps=config.optim.optimizer.eps,
            betas=config.optim.optimizer.betas,
        )
    else:
        raise NotImplementedError(f"{config.optim.optimizer.name=} not supported...")

    return optimizer


def configure_scheduler(config: DictConfig, optimizer: Optimizer) -> LRScheduler:
    if config.optim.lr_scheduler.name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.optim.lr_scheduler.factor,
            patience=config.optim.lr_scheduler.patience,
            min_lr=config.optim.lr_scheduler.min_lr,
        )
    else:
        raise NotImplementedError(f"{config.optim.lr_scheduler.name=} not supported")
    return scheduler


def dispatch_model(config: DictConfig, device: torch.device = "cpu") -> CrystalSampler:
    """Dynamic dispatch helper for configuration a model from a Hydra
    configuration.

    Parameters
    ----------
    config : DictConfig
        Hydra configuration

    Returns
    -------
    torch.nn.Module
        configured model
    """
    model_config: CrystalSamplerConfig = get_crystal_sampler_config(config)
    model = CrystalSampler(model_config, device)

    if config.model.p1_only:
        sg_logits = torch.full((230,), fill_value=-float('Inf'))
        sg_logits[0] = 1.0
        model.space_group_sampler.marginal_space_group_logits = nn.Parameter(
            sg_logits, requires_grad=False
        )

    if config.diagnostics.debug:
        for submodule in model.modules():
            submodule.register_forward_hook(nan_finder_forward_hook)

    return model


@torch.no_grad()
def log_model_weight_and_grad_norms(episode: int, model: CrystalSampler):
    # NOTE: only works distributed currently
    def weights_and_grad_norm(model: nn.Module) -> Tuple[float, float, float]:
        total_grad_norm, total_weights_norm = 0.0, 0.0
        max_grad_value = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                total_weights_norm += float(p.data.norm(2) ** 2)
                if p.grad is not None:
                    total_grad_norm += float(p.grad.data.norm(2) ** 2)
                    max_grad_value = max(max_grad_value, p.grad.data.max())
        total_weights_norm = total_weights_norm ** (1.0 / 2)
        total_grad_norm = total_grad_norm ** (1.0 / 2)
        return float(total_weights_norm), float(total_grad_norm), float(max_grad_value)

    weights_norm, grad_norm, max_grad_val = weights_and_grad_norm(model)
    wandb.log(
        {
            "Weights norm": weights_norm,
            "Gradients norm": grad_norm,
            "Max gradient value": max_grad_val,
        },
        step=episode
    )


def diagnose_and_zero_out_nan_gradients(
    model: CrystalSampler, logger: Callable, episode: int,
):
    # TODO this will break under distributed training
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if torch.any(torch.isnan(p.grad.data)):
                logger(f"Parameter {n} had a NaN gradient at step {episode:04d}.")
            p.grad.data.nan_to_num_(0.0)


def experiment_setup(config):
    set_global_embedding_tools(**config.embeddings)
    seed_everything(config.diagnostics.seed)


@torch.no_grad()
def fill_cart_atom_coords_attribute_(crystals: List[ASUCrystal]) -> None:
    frac_coords = []
    num_atoms_per_crystal = []
    lattice_lengths, lattice_angles = [], []
    for xtal in crystals:
        frac_coords.append(xtal.conventional_frac_coords)
        num_atoms_per_crystal.append(xtal.num_atoms)
        lattice_lengths.append(xtal.conventional_lattice_lengths)
        lattice_angles.append(xtal.conventional_lattice_angles)

    cart_coords: Tensor = frac_to_cart_coords(
        frac_coords=torch.cat(frac_coords, dim=0) % 1.0,
        num_atoms_per_crystal=torch.tensor(num_atoms_per_crystal, dtype=torch.long, device=xtal.device),
        lattice_lengths=torch.stack(lattice_lengths, dim=0),
        lattice_angles=torch.stack(lattice_angles, dim=0),
    )
    # (n_atoms, 3)
    for asu_cart_coords, xtal in zip(
        torch.split(cart_coords, num_atoms_per_crystal), crystals
    ):
        xtal.atom_cartesian_coordinates = asu_cart_coords


class PrioritizedExperienceReplayBuffer:
    """
    Modified from
    https://davidrpugh.github.io/stochastic-expatriate-descent/pytorch/deep-reinforcement-learning/deep-q-networks/2020/04/14/prioritized-experience-replay.html
    Fixed-size buffer to store (priority, crystal) tuples
    """

    def __init__(
        self,
        batch_size: int,
        max_buffer_size: int,
        alpha: float = 0.0,
        random_state: np.random.RandomState = None,
        storage_policy: str = "highest_priority",
    ) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
        random_state (np.random.RandomState): random number generator.
        """
        assert alpha >= 0.0
        assert storage_policy in ["most_recent", "highest_priority"]
        self._storage_policy = storage_policy
        self._batch_size = batch_size
        self._max_buffer_size = max_buffer_size
        self._current_buffer_size = 0  # current number of crystals in buffer
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state
        self._crystal_hashes: Set = set()  # for checking uniqueness
        if storage_policy == "most_recent":
            # Let buffer be a queue
            # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/deepq/replay_buffer.py#L7
            # https://github.com/ling-pan/Stochastic-GFN/blob/1f1014d630972c0662197e2db863a07e48faab4f/tfb/lib/generator/gfn.py#L133
            self._pointer = 0

        # Use numpy array as buffer for...
        # 1. storing arbitrary objects
        # 2. batched random sampling and indexing
        self._buffer = np.empty(
            self._max_buffer_size,
            dtype=[("hash", object), ("priority", float), ("crystal", ImmutableASUCrystal)],
        )

    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._current_buffer_size

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size

    @property
    def max_buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._max_buffer_size

    def add(self, priority: float, crystal: ImmutableASUCrystal) -> None:
        """Add a new crystal to memory."""
        if self._storage_policy == "highest_priority":
            self._add_highest_priority(priority, crystal)
        elif self._storage_policy == "most_recent":
            self._add_most_recent(priority, crystal)
        else:
            raise AttributeError

    def _add_most_recent(self, priority: float, crystal: ImmutableASUCrystal) -> None:
        """Only add object if hash is not in the buffer or the hash is about to
        be removed."""
        hashed_xtal: str = crystal.__hash__()
        if (
            hashed_xtal in self._crystal_hashes and
            self._buffer[self._pointer]["hash"] != hashed_xtal
        ):
            return
        else:
            if self._buffer[self._pointer]["hash"] is not None:
                self._crystal_hashes.remove(self._buffer[self._pointer]["hash"])

            self._buffer[self._pointer] = (hashed_xtal, priority, crystal)
            self._pointer = (self._pointer + 1) % self._max_buffer_size
            self._current_buffer_size = min(
                self._current_buffer_size + 1, self._max_buffer_size
            )

            self._crystal_hashes.add(hashed_xtal)

    def _add_highest_priority(self, priority: float, crystal: ImmutableASUCrystal) -> None:
        """
        Use loose hashing and only keep the highest reward version of each hash.

        PSEUDOCODE:
        if collision and new_object.reward > old_object.reward:
          # replace old_object and reward
        elif not buffer.is_full():
          # add to buffer
        else:
          if new_object.reward > buffer.smallest_reward:
              # replace lowest reward object in buffer with new_object
        """
        hashed_xtal: str = crystal.__hash__()
        collision = hashed_xtal in self._crystal_hashes
        if collision:
            # replace existing object if new one has higher priority
            colliding_idx = (hashed_xtal == self._buffer["hash"]).nonzero()
            if priority >= self._buffer[colliding_idx]["priority"]:
                self._buffer[colliding_idx] = (hashed_xtal, priority, crystal)
        elif not self.is_full():
            # add to buffer
            self._buffer[self._current_buffer_size] = (hashed_xtal, priority, crystal)
            self._current_buffer_size += 1
            self._crystal_hashes.add(hashed_xtal)
        else:
            # replace lowest priority crystal in the buffer
            idx = self._buffer["priority"].argmin()
            if priority >= self._buffer[idx]["priority"]:
                self._crystal_hashes.remove(self._buffer[idx]["hash"])
                self._buffer[idx] = (hashed_xtal, priority, crystal)
                self._crystal_hashes.add(hashed_xtal)

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._current_buffer_size == 0

    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._current_buffer_size >= self._max_buffer_size

    def sample(
        self, device: Union[str, torch.device], beta: float = 0.0
    ) -> Tuple[List[ASUCrystal], Tensor]:
        """
        Sample a batch of experiences from memory. Set beta=0 to turn off
        importance sampling.
        """
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[: self._current_buffer_size]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(
            np.arange(ps.size), size=self._batch_size, replace=True, p=sampling_probs
        )

        # select the experiences and get their priorities
        crystals: ndarray = self._buffer["crystal"][idxs]  # (self._batch_size,)
        crystals: List[ASUCrystal] = [
            xtal.to_ASUCrystal().to(device) for xtal in crystals
        ]
        priorities = torch.tensor(
            self._buffer["priority"][idxs], dtype=torch.float, device=device
        )
        # (self._batch_size,)

        # # compute importance sampling weights (for when, e.g., the objective
        # # function contains an expectation over the policy distribution)
        # weights = (self._current_buffer_size * sampling_probs[idxs]) ** -beta
        # normalized_weights = weights / weights.max()

        # return idxs, crystals, normalized_weights
        return crystals, priorities

    # def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
    #     """Update the priorities associated with particular experiences."""
    #     self._buffer["priority"][idxs] = priorities


def seed_everything(seed: int = None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def model_size(model: torch.nn.Module, as_str: bool = False) -> Union[int, str]:
    """Computes the size (in bytes) of a torch.nn.Module, including buffers.

    Parameters
    ----------
    model: torch.nn.Module
        instantiated torch model.
    as_str: bool
        whether to return the model size in string format (default: False).
    Returns
    -------
    model_size: Union[int, str]
        model size in bytes (possibly as a string).
    """
    model_size_bytes: int = 0
    for parameter in model.parameters():
        model_size_bytes += parameter.nelement() * parameter.element_size()
    for buffer in model.buffers():
        model_size_bytes += buffer.nelement() * buffer.element_size()

    if as_str:
        model_size_bytes: str = human_bytes_str(model_size_bytes)
    return model_size_bytes


def get_num_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nan_finder_forward_hook(self, inp, out):
    """
    Check for NaN inputs or outputs at each layer in the model
    Usage:
        # forward hook
        for submodule in model.modules():
            submodule.register_forward_hook(nan_hook)
    """

    outputs = isinstance(out, tuple) and out or [out]
    inputs = isinstance(inp, tuple) and inp or [inp]

    contains_nan = lambda x: torch.isnan(x).any()
    contains_inf = lambda x: torch.isinf(x).any()
    layer = self.__class__.__name__
    msg = ''

    for i, inp in enumerate(inputs):
        if isinstance(inp, Tensor):
            inp_has_nan = contains_nan(inp)
            inp_has_inf = contains_inf(inp)
            if inp_has_nan:
                msg += 'Nan'
            if inp_has_inf:
                msg += 'Inf'
            if inp_has_nan or inp_has_inf:
                raise RuntimeError(f"Found {msg} input at index: {i} in layer: {layer}")

    for i, out in enumerate(outputs):
        if isinstance(out, Tensor):
            out_has_nan = contains_nan(out)
            out_has_inf = contains_inf(out)
            if out_has_nan:
                msg += 'Nan'
            if out_has_inf:
                msg += 'Inf'
            if out_has_nan or out_has_inf:
                raise RuntimeError(f"Found {msg} output at index: {i} in layer: {layer}")


if __name__ == "__main__":
    device = "cpu"
    cesium_atomic_number = 55
    xtal = ASUCrystal(
        space_group_number=torch.tensor(1, dtype=torch.long, device=device),
        composition_space=nn.functional.one_hot(
            torch.tensor([cesium_atomic_number - 1], device=device),
            num_classes=NUM_ELEMENTS,
        ).view(-1),
        conventional_lattice_lengths=torch.tensor([6.110], device=device).expand(3),
        conventional_lattice_angles=torch.tensor([90.0], device=device).expand(3),
        atom_types=torch.zeros((0,), dtype=torch.long, device=device),
        atom_wyckoff_positions=torch.zeros(
            (0,), dtype=torch.long, device=device
        ),
        atom_asu_fractional_coordinates=torch.zeros(
            (0, 3), dtype=torch.float, device=device
        ),
        device=device,
        atom_wyckoff_shape_indices=torch.zeros(
            (0,), dtype=torch.long, device=device
        ),
    )
    xtal2 = ASUCrystal(
        space_group_number=torch.tensor(2, dtype=torch.long, device=device),
        composition_space=nn.functional.one_hot(
            torch.tensor([cesium_atomic_number - 1], device=device),
            num_classes=NUM_ELEMENTS,
        ).view(-1),
        conventional_lattice_lengths=torch.tensor([6.110], device=device).expand(3),
        conventional_lattice_angles=torch.tensor([90.0], device=device).expand(3),
        atom_types=torch.zeros((0,), dtype=torch.long, device=device),
        atom_wyckoff_positions=torch.zeros(
            (0,), dtype=torch.long, device=device
        ),
        atom_asu_fractional_coordinates=torch.zeros(
            (0, 3), dtype=torch.float, device=device
        ),
        device=device,
        atom_wyckoff_shape_indices=torch.zeros(
            (0,), dtype=torch.long, device=device
        ),
    )

    replay_buffer = PrioritizedExperienceReplayBuffer(
        batch_size=1,
        max_buffer_size=1,
        alpha=1.0,
        storage_policy="highest_priority",
    )
    fill_cart_atom_coords_attribute_([xtal, xtal2])
    replay_buffer.add(priority=1.0, crystal=xtal.to_ImmutableASUCrystal())
    print(list(replay_buffer._crystal_hashes)[0].to_ASUCrystal())
    replay_buffer.add(priority=2.0, crystal=xtal2.to_ImmutableASUCrystal())
    print(list(replay_buffer._crystal_hashes)[0].to_ASUCrystal())
