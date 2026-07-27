from abc import ABC, abstractmethod
import dataclasses

import torch.nn as nn
from torch.distributions import Categorical

from aliases import *
from utils.data_utils import lattice_params_to_matrix_torch
from crystal_classes import ASUCrystal, CrystalDict
from model.diffusion.diffusion_model import (
    EquivariantDiffusionModelConfig,
    EquivariantDiffusionModel,
)
from model.wyckoff_and_element_transformer import (
    WyckoffElementTransformer,
    WyckoffElementTransformerConfig,
)
from model.lattice_sampler import (
    TelescopingDiscreteLatticeSampler,
    LatticeSamplerConfig,
)


class BaseCrystalSampler(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample_and_log_prob_space_group(
        self, batch_size: int, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Return space_group_indices and their log probabilities."""
        raise NotImplementedError

    @abstractmethod
    def sample_and_log_prob_lattice_parameters(
        self, *args, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return lattice lengths, lattice angles, and their log probabilities."""
        raise NotImplementedError

    @abstractmethod
    def sample_and_log_prob_elements_and_wyckoffs(
        self, *args, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Return elements, Wyckoff positions, number of atoms per crystal, and
        their log probabilities.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_frac_coords(self, *args, **kwargs) -> Tensor:
        """
        Return 1 atom's fractional coordinates per Wyckoff position using the
        conventional unit cell.
        """
        raise NotImplementedError

    @abstractmethod
    def space_group_log_probs(self, space_group_indices: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def lattice_param_log_probs(
        self,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def element_and_wyckoff_log_probs(
        self,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_frac_coords_score_matching_loss(
        self,
        asu_frac_coords: Tensor,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        space_group_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
        wyckoff_shape_indices: Tensor,
        lattice_matrices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def sample_crystal(
        self,
        batch_size: int,
        diffusion_snr: float = 0.4,
        temperature: float = 1.0,
        space_group_numbers: Tensor = None,
        lattice_parameters: Tensor = None,
        get_log_probs: bool = False,  # TODO
    ) -> List[ASUCrystal]:
        with torch.no_grad():
            if space_group_numbers is None:
                # Sample space group
                (
                    space_group_indices, space_group_log_prob
                ) = self.sample_and_log_prob_space_group(batch_size, temperature)
            else:
                assert space_group_numbers.numel() == batch_size
                space_group_indices = space_group_numbers - 1
                space_group_log_prob = torch.zeros_like(space_group_indices)

            if lattice_parameters is None:
                # Sample lattice parameters
                (
                    lattice_lengths,
                    lattice_angles,
                    lattice_log_prob,
                ) = self.sample_and_log_prob_lattice_parameters(
                    space_group_indices
                )
            else:
                lattice_lengths = lattice_parameters[:, :3]
                lattice_angles = lattice_parameters[:, 3:]
                lattice_log_prob = torch.zeros(lattice_lengths.shape[0], device=lattice_lengths.device)
            lattice_matrices = lattice_params_to_matrix_torch(lattice_lengths, lattice_angles)

            # Sample elements and Wyckoff positions
            (
                element_indices,        # (n_asu_atoms,)
                wyckoff_indices,        # (n_asu_atoms,)
                n_asu_atoms_per_xtal,   # (n_crystals,)
                elements_log_prob,      # (n_asu_atoms,)
                wyckoffs_log_prob,      # (n_asu_atoms,)
                termination_log_prob,   # (n_crystals,)
            ) = self.sample_and_log_prob_elements_and_wyckoffs(
                lattice_lengths,
                lattice_angles,
                space_group_indices,
                temperature,
            )

        # Sample atom positions
        frac_coords: Tensor = self.sample_frac_coords(
            wyckoff_indices,
            element_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            snr=diffusion_snr,
            max_step_size=1e6,
        ).detach()
        device = frac_coords.device

        # Collate into crystal objects
        asu_crystals: List[ASUCrystal] = []
        atom_offsets = torch.cat(
            [
                torch.cumsum(n_asu_atoms_per_xtal, dim=0) - n_asu_atoms_per_xtal,
                n_asu_atoms_per_xtal.sum().view(1)
            ], dim=0
        )
        for i in range(len(n_asu_atoms_per_xtal)):
            asu_crystals.append(
                ASUCrystal(
                    space_group_number=1 + space_group_indices[i],
                    conventional_lattice_lengths=lattice_lengths[i],
                    conventional_lattice_angles=lattice_angles[i],
                    element_indices=element_indices[atom_offsets[i]:atom_offsets[i+1]],
                    wyckoff_indices=wyckoff_indices[atom_offsets[i]:atom_offsets[i+1]],
                    conventional_frac_coords=frac_coords[atom_offsets[i]:atom_offsets[i+1]],
                    device=device,
                )
            )
        return asu_crystals

    def compute_loss(self, crystal_dict: CrystalDict) -> Tuple[Tensor, Dict]:
        """Returns loss and a dictionary of artifacts for logging."""
        raise NotImplementedError


class SpaceGroupSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.marginal_space_group_logits: Tensor = nn.Parameter(
            torch.ones(230), requires_grad=True
        )

    def reinit_and_freeze_probs(self, space_group_probs: Tensor):
        """
        Args:
            space_group_probs: torch.float
                shape (230,). Sets the space group distribution and freezes.
        """
        device = self.marginal_space_group_logits.device
        space_group_probs = space_group_probs.to(device)
        assert (
            space_group_probs.ndim == 1 and
            space_group_probs.numel() == 230 and
            torch.all(space_group_probs >= 0) and
            torch.isclose(space_group_probs.sum(), torch.tensor(1.0, device=device))
        )
        self.marginal_space_group_logits: Tensor = nn.Parameter(
            torch.log(space_group_probs.clamp(min=1e-16)),
            requires_grad=False,
        )

    def sample_and_log_prob(self, batch_size: int = 1, temperature: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Get possibly tempered sample and its (untempered) log probability"""
        log_probs = torch.log_softmax(
            self.marginal_space_group_logits, dim=-1
        )[None, :].expand(batch_size, -1)
        # (batch_size, 230)
        sample: Tensor = Categorical(logits=log_probs / temperature).sample()
        # (batch_size,)
        sample_log_probs = log_probs.gather(dim=1, index=sample[:, None]).squeeze(dim=1)
        # (batch_size,)
        return sample, sample_log_probs

    def log_prob(self, space_group_indices: Tensor) -> Tensor:
        """
        Args:
            space_group_indices: torch.long
                shape (n_crystals,)

        Returns:
            (n_crystals,) log probabilities
        """
        normed_logits = torch.log_softmax(
            self.marginal_space_group_logits, dim=-1
        )[space_group_indices]
        # (n_crystals,)
        return normed_logits


@dataclasses.dataclass
class CrystalSamplerConfig:
    diffusion_model_config: EquivariantDiffusionModelConfig
    lattice_model_config: LatticeSamplerConfig
    transformer_config: WyckoffElementTransformerConfig

    lattice_length_noise: Optional[float] = 0.0  # Angstroms
    lattice_angle_noise: Optional[float] = 0.0  # degrees

    space_group_grad_weight: Optional[float] = 1.0
    lattice_grad_weight: Optional[float] = 1.0
    wyckoff_element_grad_weight: Optional[float] = 1.0
    frac_coord_grad_weight: Optional[float] = 1.0


class CrystalSampler(BaseCrystalSampler):
    def __init__(self, config: CrystalSamplerConfig, device: torch.device = "cpu"):
        super().__init__()
        self.config = config
        self.atom_coord_diffusion_model: nn.Module = EquivariantDiffusionModel(
            config.diffusion_model_config, device
        )
        self.space_group_sampler: nn.Module = SpaceGroupSampler()
        self.lattice_sampler: nn.Module = TelescopingDiscreteLatticeSampler(
            config.lattice_model_config
        )
        self.wyckoff_and_element_sampler: nn.Module = WyckoffElementTransformer(
            config.transformer_config,
        )

    def sample_and_log_prob_space_group(
        self, batch_size: int, temperature: float = 1.0, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Return space_group_indices and their log probabilities.

        Args:
            batch_size: number of crystals to sample
            temperature: sampling temperature

        Returns:
            space_group_indices: torch.long. Shape (batch_size,)
            log_probs: torch.float. Shape (batch_size,)
        """
        (
            space_group_indices, log_probs
        ) = self.space_group_sampler.sample_and_log_prob(batch_size, temperature)
        return space_group_indices, log_probs

    def sample_and_log_prob_lattice_parameters(
        self, space_group_indices: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return lattice lengths, lattice angles, and their log probabilities."""
        (
            lengths,        # (n_crystals, 3)
            angles,         # (n_crystals, 3)
            log_probs,      # (n_crystals,)
            regularizer,    # (n_crystals,)
        ) = self.lattice_sampler(space_group_indices)
        return lengths, angles, log_probs

    def sample_and_log_prob_elements_and_wyckoffs(
        self,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Return elements, Wyckoff positions, number of atoms per crystal, and
        their log probabilities.

        Args:
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            space_group_indices: torch.long
                shape (n_crystals,)

        Returns:
            element_indices: torch.long
                (n_asu_atoms,)
            wyckoff_indices: torch.long
                (n_asu_atoms,)
            n_asu_atoms_per_xtal: torch.long
                (n_crystals,)
            elements_log_prob: torch.float
                (n_crystals,)
            wyckoffs_log_prob: torch.float
                (n_crystals,)
            termination_log_prob: torch.float
                (n_crystals,)
        """
        (
            element_indices,        # (n_asu_atoms,)
            wyckoff_indices,        # (n_asu_atoms,)
            n_asu_atoms_per_xtal,   # (n_crystals,)
            elements_log_prob,      # (n_crystals,)
            wyckoffs_log_prob,      # (n_crystals,)
            termination_log_prob,   # (n_crystals,)
        ) = self.wyckoff_and_element_sampler.sample_and_log_prob(
            lattice_lengths,
            lattice_angles,
            space_group_indices,
            temperature,
        )
        return (
            element_indices,        # (n_asu_atoms,)
            wyckoff_indices,        # (n_asu_atoms,)
            n_asu_atoms_per_xtal,   # (n_crystals,)
            elements_log_prob,      # (n_crystals,)
            wyckoffs_log_prob,      # (n_crystals,)
            termination_log_prob,   # (n_crystals,)
        )

    def sample_frac_coords(
        self,
        wyckoff_indices: Tensor,
        element_indices: Tensor,
        space_group_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
        lattice_matrices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        snr: float = 0.4,
        max_step_size: float = 1e6,
    ) -> Tensor:
        """
        Return 1 atom's fractional coordinates per Wyckoff position using the
        conventional unit cell.

        Args:
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            element_indices: torch.long
                shape (n_asu_atoms,)
            space_group_indices: torch.long
                shape (n_crystals,)
            n_asu_atoms_per_xtal: torch.long
                shape (n_crystals,)
            lattice_matrices: torch.float
                shape (n_crystals, 3, 3)
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            snr: "Signal-to-noise" ratio. See Appendix G in
                https://arxiv.org/pdf/2011.13456.
            max_step_size:

        Returns:
            shape (n_asu_atoms, 3) fractional coordinates in conventional cell
            lattice basis.
        """
        trajectory: Tensor = self.atom_coord_diffusion_model.sample(
            wyckoff_indices,
            element_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            snr,
            max_step_size,
        )  # (num_timesteps=1000, n_asu_atoms, 3)
        return trajectory[0]  # (n_asu_atoms, 3)

    def space_group_log_probs(self, space_group_indices: Tensor) -> Tensor:
        """
        Args:
            space_group_indices: torch.long
                shape (n_crystals,)

        Returns:
            shape (n_crystals,)
        """
        return self.space_group_sampler.log_prob(space_group_indices)

    def lattice_param_log_probs(
        self,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
        noisy_lattice_lengths: Tensor = None,
        noisy_lattice_angles: Tensor = None,
    ) -> Tensor:
        (
            log_probs,      # (n_crystals,)
            regularizer,
        ) = self.lattice_sampler.log_prob(
            lattice_lengths,
            lattice_angles,
            space_group_indices,
            noisy_lattice_lengths,
            noisy_lattice_angles,
        )
        return log_probs

    def element_and_wyckoff_log_probs(
        self,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            element_indices: torch.long
                shape (n_asu_atoms,)
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            n_asu_atoms_per_xtal: torch.long
                shape (n_crystals,)
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            space_group_indices: torch.long
                shape (n_crystals,)

        Returns:
            elements_log_prob: torch.float
                shape (n_crystals,)
            wyckoff_log_prob: torch.float
                shape (n_crystals,)
            termination_log_prob: torch.float
                shape (n_crystals,)
        """
        (
            elements_log_prob,      # (n_asu_atoms,)
            wyckoff_log_prob,       # (n_asu_atoms,)
            termination_log_prob,   # (n_crystals,)
        ) = self.wyckoff_and_element_sampler.log_prob(
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            lattice_lengths,
            lattice_angles,
            space_group_indices,
        )
        return (
            elements_log_prob,      # (n_crystals,)
            wyckoff_log_prob,       # (n_crystals,)
            termination_log_prob,   # (n_crystals,)
        )

    def compute_frac_coords_score_matching_loss(
        self,
        asu_frac_coords: Tensor,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        space_group_indices: Tensor,
        n_asu_atoms_per_xtal: Tensor,
        wyckoff_shape_indices: Tensor,
        lattice_matrices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
    ) -> Tensor:
        score_matching_loss: Tensor = self.atom_coord_diffusion_model.compute_loss(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            wyckoff_shape_indices,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
        )
        return score_matching_loss

    def compute_loss(self, crystal_dict: CrystalDict) -> Tuple[Tensor, Dict]:
        """MLE on discrete variables, score matching on atom coordinates."""
        # Unpack padded dict of crystal info
        space_group_indices: Tensor = crystal_dict["space_group_indices"]
        lattice_lengths: Tensor = crystal_dict["lattice_lengths"]
        lattice_angles: Tensor = crystal_dict["lattice_angles"]
        lattice_matrices: Tensor = crystal_dict.get(
            "lattice_matrices",
            lattice_params_to_matrix_torch(lattice_lengths, lattice_angles)
        )
        padded_element_indices: Tensor = crystal_dict["element_indices"]
        padded_wyckoff_indices: Tensor = crystal_dict["wyckoff_indices"]
        n_asu_atoms_per_xtal: Tensor = crystal_dict["n_atoms_per_asu"]
        padded_wyckoff_shape_indices: Tensor = crystal_dict["wyckoff_shape_indices"]
        padded_frac_coords: Tensor = crystal_dict["frac_coords"]
        atoms_mask: Tensor = crystal_dict["atoms_mask"]
        device: torch.device = crystal_dict["device"]

        # "Unpad" stuff
        element_indices = padded_element_indices[atoms_mask]
        wyckoff_indices = padded_wyckoff_indices[atoms_mask]
        wyckoff_shape_indices = padded_wyckoff_shape_indices[atoms_mask]
        asu_frac_coords = padded_frac_coords[atoms_mask].view(-1, 3)

        if self.training and (
            self.config.lattice_length_noise > 0
            or self.config.lattice_angle_noise > 0
        ):
            # Data augmentation
            (
                noisy_lattice_lengths, noisy_lattice_angles
            ) = self.get_noisy_lattice_lengths_and_angles(
                lattice_lengths.clone(),
                lattice_angles.clone(),
                space_group_indices,
            )
        else:
            noisy_lattice_lengths = lattice_lengths
            noisy_lattice_angles = lattice_angles

        # Get space group log probability
        space_group_log_prob = self.space_group_log_probs(space_group_indices)
        # (n_crystals,)

        # Get lattice parameters log probability
        lattice_log_prob = self.lattice_param_log_probs(
            lattice_lengths,
            lattice_angles,
            space_group_indices,
            noisy_lattice_lengths,
            noisy_lattice_angles,
        )
        # (n_crystals, 6)

        # Get elements and Wyckoff positions log probability
        (
            elements_log_prob,      # (n_asu_atoms,)
            wyckoffs_log_prob,      # (n_asu_atoms,)
            termination_log_prob,   # (n_crystals,)
        ) = self.element_and_wyckoff_log_probs(
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            noisy_lattice_lengths,
            noisy_lattice_angles,
            space_group_indices,
        )

        # Get score matching loss on atomic coordinates
        score_matching_loss = self.compute_frac_coords_score_matching_loss(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            wyckoff_shape_indices,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
        )

        # Apply gradient rebalancing
        (
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
            detached_space_group_log_prob,
            detached_lattice_log_prob,
            detached_termination_log_prob,
            detached_elements_log_prob,
            detached_wyckoffs_log_prob,
            detached_score_matching_loss,
        ) = self._get_rebalanced_grads(
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
        )
        logging_artifacts = {
            "space_group_log_prob": detached_space_group_log_prob,
            "lattice_log_prob": detached_lattice_log_prob,
            "elements_log_prob": detached_elements_log_prob,
            "wyckoffs_log_prob": detached_wyckoffs_log_prob,
            "termination_log_prob": detached_termination_log_prob,
            "score_matching_loss": detached_score_matching_loss,
        }

        nll = -1.0 * (
            (space_group_log_prob + termination_log_prob).mean()  # avg over xtals
            + lattice_log_prob[lattice_log_prob != 0.0].mean()  # avg over free lattice params
            + (elements_log_prob + wyckoffs_log_prob).mean()  # avg over tokens
        )
        return nll + score_matching_loss, logging_artifacts

    @torch.no_grad()
    def compute_nll(
        self, crystal_dict: CrystalDict, num_ode_solver_steps: int = 500
    ) -> Tensor:
        # Unpack padded dict of crystal info
        space_group_indices: Tensor = crystal_dict["space_group_indices"]
        lattice_lengths: Tensor = crystal_dict["lattice_lengths"]
        lattice_angles: Tensor = crystal_dict["lattice_angles"]
        lattice_matrices: Tensor = crystal_dict.get(
            "lattice_matrices",
            lattice_params_to_matrix_torch(lattice_lengths, lattice_angles)
        )
        padded_element_indices: Tensor = crystal_dict["element_indices"]
        padded_wyckoff_indices: Tensor = crystal_dict["wyckoff_indices"]
        n_asu_atoms_per_xtal: Tensor = crystal_dict["n_atoms_per_asu"]
        padded_wyckoff_shape_indices: Tensor = crystal_dict["wyckoff_shape_indices"]
        padded_frac_coords: Tensor = crystal_dict["frac_coords"]
        atoms_mask: Tensor = crystal_dict["atoms_mask"]
        device: torch.device = crystal_dict["device"]

        # "Unpad" stuff
        element_indices = padded_element_indices[atoms_mask]
        wyckoff_indices = padded_wyckoff_indices[atoms_mask]
        wyckoff_shape_indices = padded_wyckoff_shape_indices[atoms_mask]
        asu_frac_coords = padded_frac_coords[atoms_mask].view(-1, 3)

        # Get atom coordinates' log probabilities
        frac_coords_log_prob = self.atom_coord_diffusion_model.get_log_likelihood(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            divergence_type="full",  # hutchinson
            num_ode_solver_steps=num_ode_solver_steps,
        ).detach()
        # (n_asu_atoms,)

        # Get space group log probability
        space_group_log_prob = self.space_group_log_probs(space_group_indices)
        # (n_crystals,)

        # Get lattice parameters log probability
        lattice_log_prob = self.lattice_param_log_probs(
            lattice_lengths, lattice_angles, space_group_indices,
        )
        # (n_crystals, 6)
        lattice_log_prob = lattice_log_prob.sum(dim=-1)
        # (n_crystals,)

        # Get elements and Wyckoff positions log probability
        (
            elements_log_prob,      # (n_asu_atoms,)
            wyckoffs_log_prob,      # (n_asu_atoms,)
            termination_log_prob,   # (n_crystals,)
        ) = self.element_and_wyckoff_log_probs(
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            lattice_lengths,
            lattice_angles,
            space_group_indices,
        )

        # Reduce atom attribute log probs to crystal log probs
        map_asu_atom_to_xtal = torch.repeat_interleave(
            torch.arange(space_group_indices.shape[0], device=device),
            n_asu_atoms_per_xtal,
            dim=0
        )
        # (n_crystals,)
        elements_log_prob = torch.scatter_add(
            input=torch.zeros_like(termination_log_prob),
            dim=0,
            index=map_asu_atom_to_xtal,
            src=elements_log_prob,
        )
        # (n_crystals,)
        wyckoffs_log_prob = torch.scatter_add(
            input=torch.zeros_like(termination_log_prob),
            dim=0,
            index=map_asu_atom_to_xtal,
            src=wyckoffs_log_prob,
        )
        # (n_crystals,)
        frac_coords_log_prob = torch.scatter_add(
            input=torch.zeros_like(termination_log_prob),
            dim=0,
            index=map_asu_atom_to_xtal,
            src=frac_coords_log_prob,
        )
        # (n_crystals,)

        nll = -(
            space_group_log_prob +
            lattice_log_prob +
            elements_log_prob +
            wyckoffs_log_prob +
            termination_log_prob +
            frac_coords_log_prob
        )
        return nll  # (n_crystals,)

    @torch.no_grad()
    def get_noisy_lattice_lengths_and_angles(
        self,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Noise lattice parameters subject to constraints:
            (1) Bravais lattice
            (2) Non-singular lattice matrices
            (3) Parameter values between minimum and maximum allowed values

        Args:
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            space_group_indices: torch.long
                shape (n_crystals,)

        Returns:
            noisy_lattice_lengths: torch.float
                shape (n_crystals, 3)
            noisy_lattice_angles: torch.float
                shape (n_crystals, 3)
        """
        # ---
        # Noise lattice lengths with Gaussian noise. Use rejection sampling to
        # enforce that noised lattice parameters are in the allowed ranges.
        # ---
        batch_size: int = space_group_indices.shape[0]
        device = lattice_lengths.device
        lattice_sampler: nn.Module = self.lattice_sampler
        MIN_LATTICE_LENGTH: float = self.lattice_sampler.MIN_LATTICE_LENGTH
        MAX_LATTICE_LENGTH: float = self.lattice_sampler.MAX_LATTICE_LENGTH
        MIN_LATTICE_ANGLE: float = self.lattice_sampler.MIN_LATTICE_ANGLE
        MAX_LATTICE_ANGLE: float = self.lattice_sampler.MAX_LATTICE_ANGLE

        length_transforms: Tensor = lattice_sampler.bravais_length_transforms[
            space_group_indices
        ]  # (batch_size, 3, 3)
        angle_transforms: Tensor = lattice_sampler.bravais_angle_transforms[
            space_group_indices
        ]  # (batch_size, 3, 3)
        angle_offsets: Tensor = lattice_sampler.bravais_angle_offsets[
            space_group_indices
        ]  # (batch_size, 3)

        # Get bounds for (a,b,c,alpha,beta). Bounds on gamma angle depend on
        # values of preceding lattice parameters.
        lattice_angle_bounds = torch.tensor(
            [MIN_LATTICE_ANGLE, MAX_LATTICE_ANGLE],
            dtype=torch.float,
            device=device,
        )  # (2,)  TODO: precompute
        lattice_noise_magnitude = torch.tensor(
            [
                self.config.lattice_length_noise,
                self.config.lattice_length_noise,
                self.config.lattice_length_noise,
                self.config.lattice_angle_noise,
                self.config.lattice_angle_noise,
                self.config.lattice_angle_noise,
            ],
            dtype=torch.float, device=device
        )  # (6,)  TODO: precompute

        num_samples_per_iter: int = 2
        max_iters = 100
        iteration = 0
        lattice_param_is_done = torch.tensor(
            [[False]], device=device
        ).repeat(batch_size, 6)
        accepted_noisy_lattice_parameters = torch.tensor(
            [[-1.0]], device=device
        ).repeat(batch_size, 6)
        while ~torch.all(lattice_param_is_done) and (iteration <= max_iters):
            iteration += 1
            if iteration >= max_iters:
                raise RuntimeError

            lattice_is_done = torch.all(lattice_param_is_done, dim=-1)
            # (batch_size,)
            unfinished_lengths = lattice_lengths[~lattice_is_done]
            # (n_lattices, 3)
            unfinished_angles = lattice_angles[~lattice_is_done]
            # (n_lattices, 3)
            num_lattices_left_to_noise: int = unfinished_angles.shape[0]

            # --- Noise lattice
            # Noise in [-lattice_noise_magnitude, lattice_noise_magnitude]
            noise = lattice_noise_magnitude[None, None, :] * (2.0 * torch.rand(
                num_lattices_left_to_noise, num_samples_per_iter, 6,
                device=device
            ) - 1.0)
            # (n_lattices, n_samples, 6)
            noisy_lengths = noise[:, :, :3] + unfinished_lengths[:, None, :]
            # (n_lattices, n_samples, 3)
            noisy_angles = noise[:, :, 3:] + unfinished_angles[:, None, :]
            # (n_lattices, n_samples, 3)

            # --- Enforce Bravais lattice constraints
            noisy_lengths = torch.bmm(noisy_lengths, length_transforms[~lattice_is_done])
            # (n_lattices, n_samples, 3)
            noisy_angles = (
                torch.bmm(noisy_angles, angle_transforms[~lattice_is_done])
                + angle_offsets[~lattice_is_done][:, None, :]
            )
            # (n_lattices, n_samples, 3)

            # --- Check if noised lattice parameters are valid
            lengths_are_valid = (
                (noisy_lengths >= MIN_LATTICE_LENGTH) &
                (noisy_lengths <= MAX_LATTICE_LENGTH)
            )
            # (n_lattices, n_samples, 3)

            (
                min_gamma_angle,  # (n_lattices * n_samples,)
                max_gamma_angle   # (n_lattices * n_samples,)
            ) = lattice_sampler.get_valid_gamma_angle_interval(
                alpha_and_beta_angles=noisy_angles[:, :, :2].view(-1, 2)
            )
            min_gamma_angle = min_gamma_angle + 0.01  # buffer
            max_gamma_angle = max_gamma_angle - 0.01  # buffer
            min_allowed_angles = torch.cat([
                lattice_angle_bounds[0].view(1, 1, 1).expand(
                    num_lattices_left_to_noise, num_samples_per_iter, 2
                ),
                min_gamma_angle.view(num_lattices_left_to_noise, num_samples_per_iter, 1)
            ], dim=-1)
            # (n_lattices, n_samples, 3)
            max_allowed_angles = torch.cat([
                lattice_angle_bounds[1].view(1, 1, 1).expand(
                    num_lattices_left_to_noise, num_samples_per_iter, 2
                ),
                max_gamma_angle.view(num_lattices_left_to_noise, num_samples_per_iter, 1)
            ], dim=-1)
            # (n_lattices, n_samples, 3)
            angles_are_valid = (
                (noisy_angles >= min_allowed_angles) &
                (noisy_angles <= max_allowed_angles)
            )
            # (n_lattices, n_samples, 3)

            # --- Accept first valid noised lattice parameters

            # - Accept lengths independently
            for i in range(3):  # Loop over lengths
                sample_is_valid = lengths_are_valid[:, :, i]
                # (n_lattices, n_samples)
                any_sample_is_valid = torch.any(sample_is_valid, dim=-1)
                # (n_lattices,)
                if torch.any(any_sample_is_valid):
                    # Create mask that's True if a lattice parameter should be
                    # updated by an accepted sampled length (the sample is valid
                    # and a sample is needed). Since tensor assignment does not
                    # work when applying subsequent masks, create a shape
                    # (batch_size, 6) mask to update the stored accepted lattice
                    # parameters. Could also have constructed indices with
                    # .nonzero() ops, but the code would not be as readable. See
                    #   https://stackoverflow.com/questions/77726257/torch-tensor-assignment-do-not-work-when-using-multi-mask
                    #   https://stackoverflow.com/questions/78097965/assign-values-via-two-subsequent-masking-operations-in-pytorch
                    update_lattice_params_mask = torch.nn.functional.one_hot(
                        torch.tensor([i], dtype=torch.long, device=device),
                        num_classes=6
                    ).bool().repeat(batch_size, 1)
                    # (batch_size, 6)
                    update_lattice_params_mask[lattice_is_done] = False

                    # Accept if noisy lengths are valid and none have been accepted yet
                    accept_sample_mask = (
                        any_sample_is_valid &
                        # (n_lattices,)
                        (~lattice_param_is_done[~lattice_is_done][:, i])
                        # (batch_size, 6) -> (n_lattices, 6) -> (n_lattices,)
                    )  # (n_lattices,)

                    update_lattice_params_mask[
                        (~lattice_is_done[:, None]) & update_lattice_params_mask
                        # (batch_size, 6) -> (n_lattices,)
                    ] = accept_sample_mask

                    # Get index of first accepted length
                    first_accepted_length_idx = torch.argmax(
                        sample_is_valid.float(), dim=-1
                    )[accept_sample_mask][:, None]  # (n_masked_lattices, 1)

                    # Store accepted lengths
                    accepted_noisy_lattice_parameters[update_lattice_params_mask] = (
                        # (batch_size, 6) -> -> (n_masked_lattices,)
                        noisy_lengths[:, :, i][accept_sample_mask].gather(
                            dim=1, index=first_accepted_length_idx
                        ).view(-1)
                        # (n_lattices, n_samples, 3) -> (n_lattices, n_samples)
                        # -> (n_masked_lattices, n_samples) -> (n_masked_lattices,)
                    )
                    lattice_param_is_done[update_lattice_params_mask] = True
                    # (batch_size, 6) -> (n_lattices, 6) -> (n_lattices,)
                    # -> (n_masked_lattices,)

            # - Accept angles collectively since allowable values of gamma
            #   depend on alpha and beta

            # Check if a valid set of angles has been sampled
            sample_is_valid = torch.all(angles_are_valid, dim=-1)
            # (n_lattices, n_samples)
            any_sample_is_valid = torch.any(sample_is_valid, dim=-1)
            # (n_lattices,)

            # Mask is True if a lattice parameter should be updated by an
            # accepted set of sampled angles (the set of angles is valid and
            # needed).
            update_lattice_angles_mask = torch.tensor(
                [[False, False, False, True, True, True]], device=device
            ).repeat(batch_size, 1)
            # (batch_size, 6)
            update_lattice_angles_mask[lattice_is_done] = False

            # Accept if noisy angles are valid and none have been accepted yet
            accept_sample_mask = (
                any_sample_is_valid &
                # (n_lattices,)
                torch.all(~lattice_param_is_done[~lattice_is_done][:, 3:], dim=-1)
                # (batch_size, 6) -> (n_lattices, 6) -> (n_lattices, 3) -> (n_lattices,)
            )  # (n_lattices,)

            update_lattice_angles_mask[
                (~lattice_is_done[:, None]) & update_lattice_angles_mask
                # (batch_size, 6) -> (n_lattices * 3)
            ] = accept_sample_mask[:, None].repeat(1, 3).view(-1)

            # Find index of first sample of accepted angles
            first_accepted_angles_idx = torch.argmax(
                sample_is_valid.float(), dim=-1
            )[accept_sample_mask]
            # (n_masked_lattices,)
            _lattice_idx = torch.arange(first_accepted_angles_idx.shape[0], device=device)
            # (n_masked_lattices,)

            # Store accepted angles
            accepted_noisy_lattice_parameters[update_lattice_angles_mask] = (
                # (batch_size, 6) -> (n_masked_lattices * 3)
                noisy_angles[accept_sample_mask][
                    # (n_lattices, n_samples, 3) -> (n_masked_lattices, n_samples, 3)
                    _lattice_idx, first_accepted_angles_idx
                    # -> (n_masked_lattices, 3)
                ].view(-1)
            )
            lattice_param_is_done[update_lattice_angles_mask] = True
            # (batch_size, 6) -> (n_masked_lattices * 3)

        noisy_lattice_lengths = accepted_noisy_lattice_parameters[:, :3]
        # (batch_size, 3)
        noisy_lattice_angles = accepted_noisy_lattice_parameters[:, 3:]
        # (batch_size, 3)
        return noisy_lattice_lengths, noisy_lattice_angles

    def _get_rebalanced_grads(
        self,
        space_group_log_prob: Tensor,
        lattice_log_prob: Tensor,
        termination_log_prob: Tensor,
        elements_log_prob: Tensor,
        wyckoffs_log_prob: Tensor,
        score_matching_loss: Tensor,
    ):
        """Apply straight-through estimators to rebalance gradients."""
        detached_space_group_log_prob = space_group_log_prob.detach()
        detached_lattice_log_prob = lattice_log_prob.detach()
        detached_termination_log_prob = termination_log_prob.detach()
        detached_elements_log_prob = elements_log_prob.detach()
        detached_wyckoffs_log_prob = wyckoffs_log_prob.detach()
        detached_score_matching_loss = score_matching_loss.detach()
        space_group_log_prob = (
            self.config.space_group_grad_weight * space_group_log_prob
            - self.config.space_group_grad_weight * detached_space_group_log_prob
            + detached_space_group_log_prob
        )
        lattice_log_prob = (
            self.config.lattice_grad_weight * lattice_log_prob
            - self.config.lattice_grad_weight * detached_lattice_log_prob
            + detached_lattice_log_prob
        )
        termination_log_prob = (
            self.config.wyckoff_element_grad_weight * termination_log_prob
            - self.config.wyckoff_element_grad_weight * detached_termination_log_prob
            + detached_termination_log_prob
        )
        elements_log_prob = (
            self.config.wyckoff_element_grad_weight * elements_log_prob
            - self.config.wyckoff_element_grad_weight * detached_elements_log_prob
            + detached_elements_log_prob
        )
        wyckoffs_log_prob = (
            self.config.wyckoff_element_grad_weight * wyckoffs_log_prob
            - self.config.wyckoff_element_grad_weight * detached_wyckoffs_log_prob
            + detached_wyckoffs_log_prob
        )
        score_matching_loss = (
            self.config.frac_coord_grad_weight * score_matching_loss
            - self.config.frac_coord_grad_weight * detached_score_matching_loss
            + detached_score_matching_loss
        )
        return (
            space_group_log_prob,
            lattice_log_prob,
            termination_log_prob,
            elements_log_prob,
            wyckoffs_log_prob,
            score_matching_loss,
            detached_space_group_log_prob,
            detached_lattice_log_prob,
            detached_termination_log_prob,
            detached_elements_log_prob,
            detached_wyckoffs_log_prob,
            detached_score_matching_loss,
        )


if __name__ == "__main__":
    from custom_dataset import AsymmetricUnitDataset
    from utils.embedding_utils import set_global_embedding_tools

    # Define model
    set_global_embedding_tools(
        space_group_embedding_json_path='init_tokens/space_group_features/space_group_embeddings_62dim.json',
        element_embedding_json_path='init_tokens/cgcnn_atom_init.json',
        wyckoff_embedding_json_path='init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json',
    )
    lattice_config = LatticeSamplerConfig(
        input_dimension=64,
        hidden_dimension=64,
        min_lattice_length=2.0,
        max_lattice_length=133.0,
        min_lattice_angle=60.0,
        max_lattice_angle=135.0,
        lattice_param_dim=112,
    )
    wyckoff_element_config = WyckoffElementTransformerConfig(
        hidden_dim=64,
        num_heads=2,
        num_hidden_layers=1,
        dropout_rate=0.1,
        dataset_name="mp_20",  # for lattice normalization
    )
    frac_coord_config = EquivariantDiffusionModelConfig(
        num_wn_lattice_translations=2,
        noise_scheduler_num_monte_carlo_samples=10,
        time_emb_dim=128,
        sigma_min=0.002,
        subsample_group_operations=False,
        model_type="mlp",
        # gnn_config=gnn_config,
    )
    crystal_config = CrystalSamplerConfig(
        diffusion_model_config=frac_coord_config,
        lattice_model_config=lattice_config,
        transformer_config=wyckoff_element_config,
    )
    model = CrystalSampler(crystal_config)

    # Get data
    dataset = AsymmetricUnitDataset(split="train", name="mp_20")
    crystal_dict: CrystalDict = dataset[[4, 5]]
    crystals: List[ASUCrystal] = crystal_dict.get_asu_crystal_objects()
    for crystal in crystals:
        print(crystal)

    loss, artifacts = model.compute_loss(crystal_dict)
    print(f"loss: {loss}")
    print(f"space_group_log_prob: {artifacts['space_group_log_prob']}")
    print(f"lattice_log_prob: {artifacts['lattice_log_prob']}")
    print(f"elements_log_prob: {artifacts['elements_log_prob']}")
    print(f"wyckoffs_log_prob: {artifacts['wyckoffs_log_prob']}")
    print(f"termination_log_prob: {artifacts['termination_log_prob']}")
    print(f"score matching loss: {artifacts['score_matching_loss']}")
    asu_crystals: List[ASUCrystal] = model.sample_crystal(batch_size=2)
    for xtal in asu_crystals:
        print(xtal)
