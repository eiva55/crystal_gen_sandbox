import dataclasses
import math

import torch.nn as nn
from torch.distributions.utils import logits_to_probs
from torch.distributions import MixtureSameFamily, Categorical, Beta, Dirichlet
from torch_scatter import scatter

from aliases import *
from constants import *
import global_vars
from utils.data_utils import lattice_transform_and_log_prob_mask
from model.submodules import FourierLinear, Swish


class SpaceGroupEncoder(nn.Module):
    def __init__(self, hidden_channels: int = 128, space_group_embedding_dim: int = 64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.space_group_embedding_dim = space_group_embedding_dim
        self.space_group_encoder = nn.Sequential(
            nn.Linear(
                global_vars.embedding_tools.space_group_embedding_length,
                hidden_channels
            ),
            Swish(),
            nn.Linear(hidden_channels, space_group_embedding_dim),
            Swish(),
        )

    def forward(self, space_group_indices: Tensor):
        """
        Encode space groups.

        Args:
            space_group_indices: torch.long
                shape (batch_size,)

        Returns:
            shape (batch_size, space_group_mebedding_length)
        """
        assert space_group_indices.dtype == torch.long
        space_group_embeddings = self.space_group_encoder(
            global_vars.embedding_tools.get_space_group_embedding(
                space_group_indices
            )
        )  # (batch_size, space_group_embedding_length)
        return space_group_embeddings


@dataclasses.dataclass
class LatticeSamplerConfig:
    input_dimension: int
    hidden_dimension: int
    num_hidden_layers: int = 1
    device: Optional[torch.device] = None
    use_fourier_features: Optional[bool] = True
    max_fourier_frequency: Optional[float] = 64.0
    num_fourier_frequencies: Optional[int] = 16
    lattice_lengths_transform: str = "identity"
    model_type: str = "telescoping_discrete"  # ['telescoping_discrete', 'beta_mixture']
    min_lattice_length: float = 2.0
    max_lattice_length: float = 133.0
    min_lattice_angle: float = 60.0
    max_lattice_angle: float = 135.0
    gradient_attenuation_factor: float = 1.0

    # telescoping_discrete hyperparameters
    n_bins: int = 100
    n_telescopes: int = 2
    lattice_length_bin_embedder_fourier_scale: float = 10.0
    lattice_angle_bin_embedder_fourier_scale: float = 10.0
    lattice_length_embedder_fourier_scale: float = 20.0
    lattice_angle_embedder_fourier_scale: float = 1.0
    lattice_param_dim: int = 112
    n_emb_layers: int = 4


class BaseLatticeSampler(nn.Module):
    def __init__(self, config: LatticeSamplerConfig):
        super().__init__()
        self.config = config
        self.MAX_LATTICE_LENGTH = config.max_lattice_length
        self.MIN_LATTICE_LENGTH = config.min_lattice_length
        self.MAX_LATTICE_ANGLE = config.max_lattice_angle
        self.MIN_LATTICE_ANGLE = config.min_lattice_angle

        if self.config.lattice_lengths_transform == "identity":
            self.length_transform: Callable = lambda x: x
            self.inv_length_transform: Callable = lambda x: x
        else:
            raise AttributeError

        bravais_length_transforms = []
        bravais_angle_transforms = []
        bravais_angle_offsets = []
        bravais_log_prob_masks = []
        for space_group_number in range(1, 231):
            (
                length_transform,       # (3, 3)
                angle_transform,        # (3, 3)
                angle_offset,           # (3,)
                bravais_log_prob_mask,  # (6,)
            ) = lattice_transform_and_log_prob_mask(space_group_number, "cpu")
            bravais_length_transforms.append(length_transform)
            bravais_angle_transforms.append(angle_transform)
            bravais_angle_offsets.append(angle_offset)
            bravais_log_prob_masks.append(bravais_log_prob_mask)
        bravais_length_transforms = torch.stack(bravais_length_transforms, dim=0)
        bravais_angle_transforms = torch.stack(bravais_angle_transforms, dim=0)
        bravais_angle_offsets = torch.stack(bravais_angle_offsets, dim=0)
        bravais_log_prob_masks = torch.stack(bravais_log_prob_masks, dim=0)
        self.register_buffer("bravais_length_transforms", bravais_length_transforms)
        # (num_space_groups, 3, 3)
        self.register_buffer("bravais_angle_transforms", bravais_angle_transforms)
        # (num_space_groups, 3, 3)
        self.register_buffer("bravais_angle_offsets", bravais_angle_offsets)
        # (num_space_groups, 3)
        self.register_buffer("bravais_log_prob_masks", bravais_log_prob_masks)
        # (num_space_groups, 6)

    def forward(self, *args, **kwargs):
        pass

    def log_prob(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def _get_normed_lattice_parameters(
        self,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        min_normed_param: float = -1.0,
        max_normed_param: float = 1.0,
        norm_gamma_separately: bool = True,
    ) -> Tensor:
        """
        Args:
            lattice_lengths: torch.float
                shape (batch_size, 3 lengths)
            lattice_angles: torch.float
                shape (batch_size, 3 angles)

        Returns:
            normed_lattice_parameters: torch.float
                shape (batch_size, 6 lattice parameters)
        """
        normed_param_range: float = max_normed_param - min_normed_param
        normed_lattice_lengths = (
            normed_param_range
            * (
                (self.length_transform(lattice_lengths) - self.length_transform(self.MIN_LATTICE_LENGTH))
                / (self.length_transform(self.MAX_LATTICE_LENGTH) - self.length_transform(self.MIN_LATTICE_LENGTH))
            )
            + min_normed_param
        )

        if norm_gamma_separately:
            min_gamma_angle, max_gamma_angle = self.get_valid_gamma_angle_interval(
                alpha_and_beta_angles=lattice_angles[:, :2]
            )  # (batch_size,), (batch_size,)
        else:
            min_gamma_angle, max_gamma_angle = self.MIN_LATTICE_ANGLE, self.MAX_LATTICE_ANGLE
        normed_lattice_angles = torch.cat(
            [
                normed_param_range
                * (
                    (lattice_angles[:, :2] - self.MIN_LATTICE_ANGLE)
                    / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
                )
                + min_normed_param,
                normed_param_range
                * (
                    (lattice_angles[:, -1] - min_gamma_angle)
                    / (max_gamma_angle - min_gamma_angle)
                ).unsqueeze(-1)
                + min_normed_param,
            ],
            dim=1,
        )

        normed_lattice_parameters = torch.cat(
            [normed_lattice_lengths, normed_lattice_angles], dim=1
        )
        # (batch_size, num_lattice_parameters=6)
        return normed_lattice_parameters

    def get_valid_gamma_angle_interval(self, alpha_and_beta_angles: Tensor) -> Tuple[Tensor, Tensor]:
        """
        The volume of a parallelepiped is given by
            𝑉=𝑎𝑏𝑐 * sqrt[1+2 cos𝛼 cos𝛽 cos𝛾 − (cos𝛼)^2 − (cos𝛽)^2 − (cos𝛾)^2].
        Given the first two angles, alpha and beta, we require the square root
        to be real (i.e., the lattice vectors to be non-singular). Here we
        compute the interval of gamma values which make the lattice vectors
        non-singular, and output the interval's intersection with
        [MIN_LATTICE_ANGLE, MAX_LATTICE_ANGLE].

        Args:
            alpha_and_beta_angles: shape (batch_size, 2). The 0th column gives
                alpha and the 1st column gives beta, in DEGREES.

        Returns:
            gamma_min: shape (batch_size,)
            gamma_max: shape (batch_size,)
        """
        # Compute the roots of the inside of the volume formula's square root,
        #   1+2 cos𝛼 cos𝛽 cos𝛾 − (cos𝛼)^2 − (cos𝛽)^2 − (cos𝛾)^2,
        # with respect to cos𝛾. Since the above expression is concave with
        # respect to cos𝛾, the valid interval for cos𝛾 is between the roots.
        # With a little algebra, the roots are given as:
        #   cos𝛾 = cos𝛼 cos𝛽 +/- 0.5 * sqrt[4 * (cos𝛼)^2 * (cos𝛽)^2 - 4 * ((cos𝛼)^2 + (cos𝛽)^2 - 1)]

        cos_alpha = torch.cos(alpha_and_beta_angles[:, 0] * torch.pi / 180.0)  # (batch_size,)
        cos_beta = torch.cos(alpha_and_beta_angles[:, 1] * torch.pi / 180.0)  # (batch_size,)

        cos_alpha_sq = cos_alpha**2  # (batch_size,)
        cos_beta_sq = cos_beta**2  # (batch_size,)

        term1 = cos_alpha * cos_beta  # (batch_size,)
        term2 = 0.5 * torch.sqrt(
            4 * cos_alpha_sq * cos_beta_sq - 4 * (cos_alpha_sq + cos_beta_sq - 1)
        )  # (batch_size,)

        # arccos decreases monotonically
        gamma_min = (
            torch.acos((term1 + term2).clamp(min=-1.0, max=1.0)) * 180.0 / torch.pi
        )  # (batch_size,)
        gamma_max = (
            torch.acos((term1 - term2).clamp(min=-1, max=1.0)) * 180.0 / torch.pi
        )  # (batch_size,)

        # Note: gamma_min == gamma_max iff (alpha or beta = pi * n) for an
        # integer n (i.e., a multiple of 180 degrees), which is impossible.
        return (
            gamma_min.clamp(min=self.MIN_LATTICE_ANGLE, max=self.MAX_LATTICE_ANGLE),
            gamma_max.clamp(min=self.MIN_LATTICE_ANGLE, max=self.MAX_LATTICE_ANGLE),
        )

    def validate_arguments(self, chemistry: Tensor, space_groups: Tensor) -> None:
        batch_size: int = chemistry.shape[0]
        assert space_groups.shape[0] == batch_size and (
            space_groups.ndim == 1 or space_groups.shape[1] == 1
        ), f"Expected space group identifiers of shape (B, 1), but got: {space_groups.shape=}"
        assert torch.logical_and(space_groups >= 0, space_groups <= 229).all()


class TelescopingDiscreteLatticeSampler(BaseLatticeSampler):
    """Agnostic to composition space and does not use GFlowTrace."""
    def __init__(self, config: LatticeSamplerConfig):
        super().__init__(config)
        self.n_bins: int = self.config.n_bins  # number of bins per telescope
        self.n_telescopes = self.config.n_telescopes
        self.min_bin_edge = -4.0
        self.max_bin_edge = 4.0

        # Condition lattice on space group and composition space
        self.space_group_encoder = SpaceGroupEncoder(
            hidden_channels=256,
            space_group_embedding_dim=self.config.input_dimension,
        )

        self.lattice_param_dim = self.config.lattice_param_dim
        # Tuned to differentiate 0.02 Ang differences when raw lattice lengths
        # are shifted and normalized into [-4, 4]
        self.lattice_length_embedder: nn.Module = nn.Sequential(
            FourierLinear(
                input_dim=1,
                num_fourier_frequencies=128,
                scale=self.config.lattice_length_embedder_fourier_scale,
                num_layers=self.config.n_emb_layers,
                output_dim=512,
                use_bias=True,
            ),  # No activation after skip connections
            nn.Linear(512, self.lattice_param_dim),  # Project down to more manageable size
            nn.SiLU(),
        )

        # Tuned to differentiate 3 degree differences when raw lattice angles
        # are shifted and normalized into [-4, 4]
        self.lattice_angle_embedder: nn.Module = nn.Sequential(
            FourierLinear(
                input_dim=1,
                num_fourier_frequencies=128,
                scale=self.config.lattice_angle_embedder_fourier_scale,
                num_layers=self.config.n_emb_layers,
                output_dim=256,
                use_bias=True,
            ),  # No activation after skip connections
            nn.Linear(256, self.lattice_param_dim),  # Project down to more manageable size
            nn.SiLU(),
        )
        # Embed bins, each represented as 2 bin edges
        self.length_bin_embedder = nn.Sequential(
            FourierLinear(
                input_dim=2,
                num_fourier_frequencies=128,
                scale=self.config.lattice_length_bin_embedder_fourier_scale,
                output_dim=256,
                num_layers=self.config.n_emb_layers,
                use_bias=True,
            ),
            nn.Linear(256, self.config.hidden_dimension),
            nn.SiLU(),
        )
        self.angle_bin_embedder = nn.Sequential(
            FourierLinear(
                input_dim=2,
                num_fourier_frequencies=128,
                scale=self.config.lattice_angle_bin_embedder_fourier_scale,
                output_dim=256,
                num_layers=self.config.n_emb_layers,
                use_bias=True,
            ),
            nn.Linear(256, self.config.hidden_dimension),
            nn.SiLU(),
        )
        self.bin_conditioning_info_dim = (
            self.config.input_dimension + 6 * self.lattice_param_dim + 6
        )
        # space_group_and_chem_emb_size + lattice_emb_size + lattice_mask_size
        self.bin_logit_head = nn.Sequential(
            nn.Linear(
                self.config.hidden_dimension + self.bin_conditioning_info_dim,
                self.config.hidden_dimension
            ),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dimension, 6),
        )
        self.register_buffer("grid_pts", torch.linspace(0, 1, self.n_bins + 1))

        # -- Store discretize(norm(Bravais lattice angle constraints))
        angle_offsets = self.bravais_angle_offsets.clone()
        unconstrained_angle_mask = (angle_offsets == 0.0)
        # (num_space_groups, 3)
        angle_offsets[unconstrained_angle_mask] = self.MIN_LATTICE_ANGLE  # dummy
        _normed_params = self._get_normed_lattice_parameters(
            self.MIN_LATTICE_LENGTH * torch.ones_like(self.bravais_angle_offsets),  # dummy
            angle_offsets,
            self.min_bin_edge,
            self.max_bin_edge,
            norm_gamma_separately=False,
        )  # (num_space_groups, 6)
        _discretized_normed_params = self.get_discretized_normed_lattice_params(
            _normed_params, angle_offsets[:, :2]
        )  # (num_space_groups, 6)
        _discretized_normed_angle_offsets = _discretized_normed_params[:, 3:]
        _discretized_normed_angle_offsets[unconstrained_angle_mask] = 0.0
        self.register_buffer(
            "discretized_normed_bravais_angle_offsets",
            _discretized_normed_angle_offsets
        )  # (num_space_groups, 3)

    def forward(self, space_group_indices: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample 6 lattice parameters constrained by the Bravais lattices.

        Args:
            space_group_indices: torch.long
                shape (batch_size,)

        Returns:
            lengths: torch.float
                shape (batch_size, 3)
            angles: torch.float
                shape (batch_size, 3)
            log_pfs: torch.float
                shape (batch_size,)
            regularizer: torch.float
                shape (batch_size,)
        """
        device = space_group_indices.device
        batch_size = space_group_indices.shape[0]
        sg_features = self.space_group_encoder(space_group_indices)
        # (batch_size, self.config.input_dimension)

        num_angles = 3
        num_lengths = 3
        num_lattice_parameters = num_angles + num_lengths

        # Make tensors for transforming from normed to raw lattice parameters
        lattice_params_transform = torch.cat(
            (
                torch.tensor(
                    [
                        self.length_transform(self.MAX_LATTICE_LENGTH)
                        - self.length_transform(self.MIN_LATTICE_LENGTH)
                    ], device=device,
                ).expand(batch_size, num_lengths),
                torch.tensor(
                    [self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE], device=device
                ).expand(batch_size, num_angles),
            ),
            dim=1,
        )  # (batch_size, num_lattice_parameters)
        lattice_params_offset = torch.cat(
            (
                torch.tensor(
                    [self.length_transform(self.MIN_LATTICE_LENGTH)], device=device
                ).expand(batch_size, num_lengths),
                torch.tensor(
                    [self.MIN_LATTICE_ANGLE], device=device
                ).expand(batch_size, num_angles),
            ),
            dim=1,
        )  # (batch_size, num_lattice_parameters)

        # Sample normalized lattice params (in [-4,4] so NN outputs are stable)
        normed_lattice_parameters = torch.zeros(batch_size, num_lattice_parameters, device=device)
        current_lattice_embedding = torch.zeros(
            batch_size, 6 * self.lattice_param_dim, device=device
        )
        log_pfs = []  # list of tensors each of shape (batch_size,)
        lattice_mask = torch.zeros(6, device=device)
        regularizer = torch.zeros(batch_size, device=device)
        alpha_and_beta_angles = None
        bravais_length_transforms = self.bravais_length_transforms[space_group_indices]
        # (batch_size, 3, 3)
        bravais_angle_transforms = self.bravais_angle_transforms[space_group_indices]
        # (batch_size, 3, 3)
        discretized_normed_bravais_angle_offsets = (
            self.discretized_normed_bravais_angle_offsets[space_group_indices]
        )  # (batch_size, 3)
        for i in range(num_lattice_parameters):
            # Get features to condition the bin logits on
            if i > 0:
                # Embed the lattice parameter that was just sampled
                if i < 4:
                    current_lattice_embedding[
                        :, self.lattice_param_dim * (i-1):self.lattice_param_dim * i
                    ] = self.lattice_length_embedder(
                        normed_lattice_parameters[:, i-1].view(-1, 1)
                    )  # (batch_size, lattice_param_emb=64)
                else:
                    current_lattice_embedding[
                        :, self.lattice_param_dim * (i-1): self.lattice_param_dim * i
                    ] = self.lattice_angle_embedder(
                        normed_lattice_parameters[:, i-1].view(-1, 1)
                    )  # (batch_size, lattice_param_emb=64)
            current_state_features = torch.cat(
                [
                    sg_features,
                    current_lattice_embedding,
                    lattice_mask[None, :].expand(batch_size, 6)
                ], dim=1
            )[:, None, :].expand(-1, self.n_bins, -1)
            # (batch_size, n_bins, embedding_dim)
            if i == 5:
                alpha_and_beta_angles = (
                    (normed_lattice_parameters[:, 3:5] - self.min_bin_edge)
                    / (self.max_bin_edge - self.min_bin_edge)
                ) * lattice_params_transform[:, 3:5] + lattice_params_offset[:, 3:5]
            sample, log_prob = self._sample_and_log_prob(
                lattice_param_index=i,
                bin_embedder=self.length_bin_embedder if i < 3 else self.angle_bin_embedder,
                batch_size=batch_size,
                device=device,
                z=current_state_features,
                alpha_and_beta_angles=alpha_and_beta_angles,
                enforce_gamma_bounds=i == (num_lattice_parameters - 1),
            )  # (batch_size,), (batch_size,)

            normed_lattice_parameters[:, i] = sample
            lattice_mask[i] = 1.0
            log_pfs.append(log_prob)

            # Apply Bravais lattice constraints at each autoregressive step so
            # that the next step is conditioned on the correct thing
            if i < 3:  # Constrain lengths
                normed_lattice_parameters[:, :3] = torch.bmm(
                    normed_lattice_parameters[:, :3][:, None, :],
                    bravais_length_transforms
                ).squeeze(dim=1)
            else:  # Constrain angles
                normed_lattice_parameters[:, 3:] = torch.bmm(
                    normed_lattice_parameters[:, 3:][:, None, :],
                    bravais_angle_transforms
                ).squeeze(dim=1) + discretized_normed_bravais_angle_offsets

            # TODO: update regularizer if we need that later

        log_pfs = torch.stack(log_pfs, dim=1)
        # (batch_size, num_lattice_parameters)

        # --
        # Transform lattice_parameters from [min_bin_edge, max_bin_edge] to
        # [0, 1] to [MIN_PARAM, MAX_PARAM]
        # --
        lattice_parameters = (
            (normed_lattice_parameters - self.min_bin_edge)
            / (self.max_bin_edge - self.min_bin_edge)
        ) * lattice_params_transform + lattice_params_offset
        # (batch_size, num_lattice_parameters)

        lengths: Tensor = self.inv_length_transform(lattice_parameters[:, :num_lengths])
        # (batch_size, 3)
        angles: Tensor = lattice_parameters[:, num_lengths:]
        # (batch_size, 3)

        # Mask out steps which were constrained by the Bravais lattice.
        # Mask assumes lattice parameters ordered as (a,b,c,alpha,beta,gamma)
        log_pf_masks: Tensor = self.bravais_log_prob_masks[space_group_indices]
        # (batch_size, num_lattice_parameters)
        log_pfs = (log_pfs * log_pf_masks).sum(dim=1)
        # (batch_size,)
        if self.config.gradient_attenuation_factor != 1.0:
            log_pfs_detach = log_pfs.detach()
            log_pfs = (
                self.config.gradient_attenuation_factor * log_pfs
                - self.config.gradient_attenuation_factor * log_pfs_detach
                + log_pfs_detach
            )
        return lengths, angles, log_pfs, regularizer

    def log_prob(
        self,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        space_group_indices: Tensor,
        noisy_lattice_lengths: Tensor = None,
        noisy_lattice_angles: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """Given space groups, get the forward probability that 'self' samples
        'lattice_lengths' and 'lattice_angles'.

        Args:
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)
            space_group_indices: torch.long
                shape (n_crystals,)
            noisy_lattice_lengths: torch.float
                shape (n_crystals, 3)
            noisy_lattice_angles: torch.float
                shape (n_crystals, 3)

        Returns:
            log_forward_probability: shape (batch_size,) tensor with the forward
                probability of sampling 'lattice_lengths' and 'lattice_angles'.
            regularizer: shape (batch_size,) tensor
        """
        batch_size: int = lattice_lengths.shape[0]
        num_lengths = num_angles = 3
        num_lattice_parameters = num_lengths + num_angles

        # - Train on GT lattice, condition on noisy lattice
        noisy_lattice_lengths: Tensor = (
            noisy_lattice_lengths
            if self.training and noisy_lattice_lengths is not None
            else lattice_lengths
        )  # (num_crystals, 3)
        noisy_lattice_angles: Tensor = (
            noisy_lattice_angles
            if self.training and noisy_lattice_angles is not None
            else lattice_angles
        )  # (num_crystals, 3)
        assert (
            lattice_lengths.shape == lattice_angles.shape
            == noisy_lattice_lengths.shape == noisy_lattice_angles.shape
            == (batch_size, 3)
        )
        device = lattice_lengths.device

        # -- Get input to lattice sampler for crystals without lattice parameters
        sg_features = self.space_group_encoder(space_group_indices)
        # (batch_size, D=space_group_embedding_length + chemistry_embedding_length)

        # -- Get log forward probability of all 6 lattice parameters

        # - Normalize lattice to [min_bin_edge, max_bin_edge]
        normed_lattice_parameters: Tensor = self._get_normed_lattice_parameters(
            lattice_lengths,
            lattice_angles,
            self.min_bin_edge,
            self.max_bin_edge,
            norm_gamma_separately=False,
        )  # (batch_size, num_lattice_parameters=6)
        normed_noisy_lattice_parameters: Tensor = self._get_normed_lattice_parameters(
            noisy_lattice_lengths,
            noisy_lattice_angles,
            self.min_bin_edge,
            self.max_bin_edge,
            norm_gamma_separately=False,
        ) if self.training else normed_lattice_parameters
        # (batch_size, num_lattice_parameters=6)

        # - Discretize the lattice
        _normed_gt_and_noisy_lattice_params: Tensor = self.get_discretized_normed_lattice_params(
            torch.cat([normed_lattice_parameters, normed_noisy_lattice_parameters], dim=0),
            torch.cat([lattice_angles[:, 1:], noisy_lattice_angles[:, 1:]], dim=0)
        )
        normed_lattice_parameters = _normed_gt_and_noisy_lattice_params[:batch_size]
        normed_noisy_lattice_parameters = _normed_gt_and_noisy_lattice_params[batch_size:]
        # (batch_size, num_lattice_parameters=6)
        # --

        log_pfs = []  # Order is reversed from the forward trajectory
        regularizer = torch.zeros(batch_size, device=device)
        lattice_mask = torch.ones(6, device=device)
        current_lattice_embedding = torch.cat(
            [
                self.lattice_length_embedder(
                    normed_noisy_lattice_parameters[:, :3].view(-1, 3, 1)
                ).view(batch_size, -1),  # (batch_size, 3*D, 1) -> (batch_size, 3*D)
                self.lattice_angle_embedder(
                    normed_noisy_lattice_parameters[:, 3:].view(-1, 3, 1)
                ).view(batch_size, -1)  # (batch_size, 3*D, 1) -> (batch_size, 3*D)
            ], dim=-1
        )  # (batch_size, 6*D)
        for i in range(num_lattice_parameters - 1, -1, -1):  # Loop over lattice parameters to delete
            # - Remove lattice parameter
            next_normed_lattice_parameter = normed_lattice_parameters[:, i].clone()
            # (batch_size,)
            current_lattice_embedding[
                :, self.lattice_param_dim * i: self.lattice_param_dim * (i+1)
            ] = 0.0

            # Mask non-existent lattice parameters. We will not apply the mask
            # with a Hadamard product since 0 represents a lattice parameter.
            # Instead, concatenate mask indicating which parameters are present.
            lattice_mask[i] = 0.0  # (6,)

            current_state_features = torch.cat(
                [
                    sg_features,
                    current_lattice_embedding,
                    lattice_mask[None, :].expand(batch_size, 6)
                ], dim=1
            )[:, None, :].expand(-1, self.n_bins, -1)
            # (batch_size, n_bins, embedding_dim)
            _, log_prob = self._sample_and_log_prob(
                lattice_param_index=i,
                bin_embedder=self.length_bin_embedder if i < 3 else self.angle_bin_embedder,
                batch_size=batch_size,
                device=device,
                x=next_normed_lattice_parameter,
                z=current_state_features,
            )  # (batch_size,), (batch_size,)
            log_pfs.append(log_prob)

            # TODO: update regularizer if we need that later

        # Probabilities of sampling the lattice parameters, in reverse order
        log_pfs = torch.stack(log_pfs, dim=1)
        # (batch_size, num_lattice_parameters)

        # Probabilities in forward order
        column_indices_reversed = torch.arange(
            start=log_pfs.shape[1] - 1, end=-1, step=-1, dtype=torch.long, device=device
        )
        log_pfs = log_pfs[:, column_indices_reversed]

        # -- Mask out log forward probabilities according to the Bravais lattice
        log_pf_masks = self.bravais_log_prob_masks[space_group_indices]
        # (batch_size, num_lattice_parameters)
        log_pfs = log_pfs * log_pf_masks
        # (batch_size, num_lattice_parameters)
        if self.config.gradient_attenuation_factor != 1.0:
            log_pfs_detach = log_pfs.detach()
            log_pfs = (
                self.config.gradient_attenuation_factor * log_pfs
                - self.config.gradient_attenuation_factor * log_pfs_detach
                + log_pfs_detach
            )

        return log_pfs, regularizer

    def _sample_and_log_prob(
        self,
        lattice_param_index: int,
        bin_embedder: nn.Module,
        batch_size: int,
        device: torch.device,
        x: Tensor = None,
        z: Tensor = None,
        alpha_and_beta_angles: Tensor = None,
        enforce_gamma_bounds: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Implements the telescoping discrete sampling and inference.

        Args:
            lattice_param_index: int
                In [0, 5].
            bin_embedder: nn.Module
                Either self.length_bin_embedder or self.angle_bin_embedder.
            x: torch.float
                shape (batch_size,) Tensor of points in
                [min_bin_edge, max_bin_edge] to bin and get model log_prob of.
                If None, will sample from the model.
            z: torch.float
                shape (batch_size, n_bins, self.bin_conditioning_info_dim)
                Tensor of conditioning information. If None, sets to zeroes.
            alpha_and_beta_angles: torch.float
                shape (batch_size, 2) alpha and beta lattice angles in degrees.
                Only used if 'enforce_gamma_bounds' is True.
            enforce_gamma_bounds: bool
                If True, set the probability of bins outside the gamma bounds to
                zero. Ensures lattices are non-singular during forward sampling.

        Returns:
            samples: torch.float
                shape (batch_size,)
            log_probs: torch.float
                shape (batch_size,)
        """
        assert 0 <= lattice_param_index < 6
        if x is not None:
            assert torch.all((x >= self.min_bin_edge) & (x <= self.max_bin_edge))
        if z is None:
            z = torch.zeros(
                batch_size, self.n_bins, self.bin_conditioning_info_dim,
                device=device
            )
        if enforce_gamma_bounds:
            assert (
                alpha_and_beta_angles is not None and
                alpha_and_beta_angles.shape == (batch_size, 2)
            )
            min_gamma, max_gamma = self.get_valid_gamma_angle_interval(
                alpha_and_beta_angles
            )  # (batch_size,), (batch_size,)
            min_gamma, max_gamma = min_gamma.view(-1, 1), max_gamma.view(-1, 1)
            # (batch_size, 1), (batch_size, 1)

            # Normalize from [MIN_ANGLE, MAX_ANGLE] to [0, 1] to [min_bin_edge,
            # max_bin_edge]
            min_normed_gamma: Tensor = (self.max_bin_edge - self.min_bin_edge) * (
                (min_gamma - self.MIN_LATTICE_ANGLE)
                / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
            ) + self.min_bin_edge
            max_normed_gamma: Tensor = (self.max_bin_edge - self.min_bin_edge) * (
                (max_gamma - self.MIN_LATTICE_ANGLE)
                / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
            ) + self.min_bin_edge

        _batch_idxs = torch.arange(batch_size, device=device)
        min_bin_edge = self.min_bin_edge * torch.ones(batch_size, device=device)
        max_bin_edge = self.max_bin_edge * torch.ones(batch_size, device=device)
        log_probs = torch.zeros(batch_size, device=device)
        for j in range(self.n_telescopes):
            bin_edges = (
                min_bin_edge[:, None] + (max_bin_edge - min_bin_edge)[:, None]
                * self.grid_pts[None, :].expand(batch_size, -1)
            )  # (batch_size, n_bins + 1)
            bins = torch.stack([bin_edges[:, :-1], bin_edges[:, 1:]], dim=-1)
            # (batch_size, n_bins, 2)

            bin_logits = self.bin_logit_head(
                torch.cat([bin_embedder(bins), z], dim=-1)
            )[..., lattice_param_index]
            # (batch_size, n_bins)
            if enforce_gamma_bounds:
                if j < self.n_telescopes - 1:
                    # Bin is invalid if (bin right edge < min_normed_gamma) or
                    # (bin left edge > max_normed_gamma)

                    # Enforce that a valid bin contains > 1 valid sub-bin
                    epsilon = (bin_edges[0, 1] - bin_edges[0, 0]) / self.n_bins

                    zero_prob_bins_mask = (
                        (bins[:, :, 1] - epsilon < min_normed_gamma) |
                        (bins[:, :, 0] + epsilon > max_normed_gamma)
                    )
                else:
                    # Bin is invalid if (bin midpoint < min_normed_gamma) or
                    # (bin midpoint > max_normed_gamma)
                    bin_midpoints = torch.mean(bins, dim=-1)
                    # (batch_size, n_bins)
                    zero_prob_bins_mask = (
                        (bin_midpoints < min_normed_gamma) |
                        (bin_midpoints > max_normed_gamma)
                    )  # (batch_size, n_bins)

                if torch.any(torch.all(zero_prob_bins_mask, dim=-1)):
                    raise NotImplementedError("All bins were invalid")
                bin_logits = (
                    bin_logits +
                    zero_prob_bins_mask.float().masked_fill(
                        zero_prob_bins_mask, -float('inf')
                    )
                )

            if x is None:
                dist = Categorical(logits=bin_logits)
                bin_idxs = dist.sample()
                # (batch_size,)
            else:
                bin_idxs = torch.bucketize(
                    (x - min_bin_edge) / (max_bin_edge - min_bin_edge),
                    self.grid_pts[1:]
                )  # (batch_size,)

            log_probs = log_probs + torch.nn.functional.log_softmax(
                bin_logits, dim=-1
            ).gather(index=bin_idxs.view(-1, 1), dim=-1).view(-1)
            # (batch_size,)

            chosen_bins = bins[_batch_idxs, bin_idxs]  # (batch_size, 2)
            min_bin_edge = chosen_bins[:, 0]  # (batch_size,)
            max_bin_edge = chosen_bins[:, 1]  # (batch_size,)

        # Return the midpoint of each bin
        samples = torch.mean(chosen_bins, dim=-1)
        # (batch_size,)
        return samples, log_probs

    @torch.no_grad()
    def probs(self, crystals_dict: Dict, normed_grid_pts: Tensor) -> Tensor:
        """
        For visualizing the probability distribution.

        Args:
            crystals_dict:
            normed_grid_pts: shape (n_grid_pts, 6)

        Returns:
            log_pf: shape (batch_size, num_lattice_parameters=6, n_bins)
        """
        assert torch.all((normed_grid_pts >= self.min_bin_edge) & (normed_grid_pts <= self.max_bin_edge))

        def _grid_log_probs(
            lattice_param_index: int,
            bin_embedder: nn.Module,
            device: torch.device,
            x: Tensor,
            z: Tensor,
        ):
            batch_size = 1
            assert 0 <= lattice_param_index < 6
            if x is not None:
                assert torch.all((x >= self.min_bin_edge) & (x <= self.max_bin_edge))
            if z is None:
                z = torch.zeros(
                    batch_size, self.n_bins, self.bin_conditioning_info_dim,
                    device=device
                )

            zeros = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            min_bin_edge = self.min_bin_edge * torch.ones(batch_size, device=device)
            max_bin_edge = self.max_bin_edge * torch.ones(batch_size, device=device)
            log_probs = torch.zeros(x.shape[0], device=device)
            for j in range(self.n_telescopes):
                bin_edges = (
                    min_bin_edge[:, None] + (max_bin_edge - min_bin_edge)[:, None]
                    * self.grid_pts[None, :].expand(min_bin_edge.shape[0], -1)
                )  # (n, n_bins + 1)
                bins = torch.stack([bin_edges[:, :-1], bin_edges[:, 1:]], dim=-1)
                # (n, n_bins, 2)

                bin_logits = self.bin_logit_head(
                    torch.cat(
                        [
                            bin_embedder(bins),
                            z.expand(bins.shape[0], self.n_bins, -1)
                        ], dim=-1
                    )
                )[..., lattice_param_index]
                # (n, n_bins)
                bin_idxs = torch.bucketize(
                    (x - min_bin_edge) / (max_bin_edge - min_bin_edge),
                    self.grid_pts[1:]
                )  # (n_grid_pts,)

                log_probs = log_probs + torch.nn.functional.log_softmax(
                    bin_logits, dim=-1
                ).gather(index=bin_idxs.view(1, -1), dim=-1).view(-1)
                # (n_grid_pts,)

                chosen_bins = bins[zeros, bin_idxs]  # (n_grid_pts, 2)
                min_bin_edge = chosen_bins[:, 0]  # (n_grid_pts,)
                max_bin_edge = chosen_bins[:, 1]  # (n_grid_pts,)

            return log_probs

        batch_size: int = crystals_dict["n_atoms_per_asu"].shape[0]
        num_lengths = num_angles = 3
        num_lattice_parameters = num_lengths + num_angles
        lattice_lengths: Tensor = crystals_dict["lattice_lengths"]
        # (batch_size, 3)
        lattice_angles: Tensor = crystals_dict["lattice_angles"]
        # (batch_size, 3)
        space_group_indices: Tensor = crystals_dict["space_group_indices"]
        # (batch_size,)
        device = crystals_dict["device"]
        log_pf_masks = self.bravais_log_prob_masks[space_group_indices].view(-1)
        # (num_lattice_parameters,)

        # -- Get input to lattice sampler for crystals without lattice parameters
        sg_features = self.space_group_encoder(space_group_indices)
        # (batch_size, D=space_group_embedding_length + chemistry_embedding_length)

        # -- Get log forward probability of all 6 lattice parameters

        # - Normalize lattice to [min_bin_edge, max_bin_edge]
        normed_lattice_parameters: Tensor = self._get_normed_lattice_parameters(
            lattice_lengths,
            lattice_angles,
            self.min_bin_edge,
            self.max_bin_edge,
            norm_gamma_separately=False,
        )  # (batch_size, num_lattice_parameters=6)

        # - Discretize the lattice
        normed_lattice_parameters = self.get_discretized_normed_lattice_params(
            normed_lattice_parameters, lattice_angles[:, 1:]
        )

        lattice_mask = torch.ones(6, device=device)
        current_lattice_embedding = torch.cat(
            [
                self.lattice_length_embedder(
                    normed_lattice_parameters[:, :3].view(-1, 3, 1)
                ).view(batch_size, -1),  # (batch_size, 3*D, 1) -> (batch_size, 3*D)
                self.lattice_angle_embedder(
                    normed_lattice_parameters[:, 3:].view(-1, 3, 1)
                ).view(batch_size, -1)  # (batch_size, 3*D, 1) -> (batch_size, 3*D)
            ], dim=-1
        )  # (batch_size, 6*D)

        all_log_probs = []
        for i in range(num_lattice_parameters - 1, -1, -1):  # Loop over lattice parameters to delete
            grid_pts = normed_grid_pts[:, i]

            # - Remove lattice parameter
            next_normed_lattice_parameter = normed_lattice_parameters[:, i].clone()
            # (batch_size,)
            current_lattice_embedding[
                :, self.lattice_param_dim * i: self.lattice_param_dim * (i+1)
            ] = 0.0

            # Mask non-existent lattice parameters. We will not apply the mask
            # with a Hadamard product since 0 represents a lattice parameter.
            # Instead, concatenate mask indicating which parameters are present.
            lattice_mask[i] = 0.0  # (6,)

            current_state_features = torch.cat(
                [
                    sg_features,
                    current_lattice_embedding,
                    lattice_mask[None, :].expand(batch_size, 6)
                ], dim=1
            )[:, None, :].expand(1, self.n_bins, -1)
            # (batch_size, n_bins, embedding_dim)

            log_probs = _grid_log_probs(
                lattice_param_index=i,
                bin_embedder=self.length_bin_embedder if i < 3 else self.angle_bin_embedder,
                device=device,
                x=torch.cat([next_normed_lattice_parameter, grid_pts], dim=0),
                z=current_state_features,
            )  # (1 + n_grid_pts,)

            all_log_probs.append(log_probs * log_pf_masks[i])

        # Probabilities of sampling the lattice parameters, in reverse order
        log_pfs = torch.stack(all_log_probs, dim=0)
        # (num_lattice_parameters, 1 + n_grid_pts)

        row_indices_reversed = torch.arange(
            start=log_pfs.shape[0] - 1, end=-1, step=-1, dtype=torch.long, device=device
        )
        log_pfs = log_pfs[row_indices_reversed, :]

        return log_pfs.exp()

    @torch.no_grad()
    def get_discretized_normed_lattice_params(
        self,
        normed_lattice_parameters: Tensor,
        raw_alpha_and_beta_angles: Tensor,
    ) -> Tensor:
        """
        Args:
            normed_lattice_parameters: torch.float
                shape (batch_size, 6) lattice parameters normalized into
                [min_bin_edge, max_bin_edge]
            raw_alpha_and_beta_angles: torch.float
                shape (batch_size, 2) alpha and beta lattice angles in degrees.

        Returns:
            discretized_lattice_parameters: torch.float
                shape (batch_size, 6). The first 3 columns are conventional
                lattice lengths, while the last 3 are angles.
        """
        assert torch.all(
            (normed_lattice_parameters >= self.min_bin_edge) &
            (normed_lattice_parameters <= self.max_bin_edge)
        )
        batch_size = normed_lattice_parameters.shape[0]
        device = normed_lattice_parameters.device
        _batch_idxs = torch.arange(batch_size, device=device)

        discretized_normed_lattice_parameters = torch.zeros(batch_size, 6, device=device)
        for i in range(6):
            min_bin_edge = self.min_bin_edge * torch.ones(batch_size, device=device)
            max_bin_edge = self.max_bin_edge * torch.ones(batch_size, device=device)
            x = normed_lattice_parameters[:, i]
            # (batch_size,)
            for j in range(self.n_telescopes):
                bin_edges = (
                    min_bin_edge[:, None] + (max_bin_edge - min_bin_edge)[:, None]
                    * self.grid_pts[None, :].expand(batch_size, -1)
                )  # (batch_size, n_bins + 1)
                bins = torch.stack([bin_edges[:, :-1], bin_edges[:, 1:]], dim=-1)
                # (batch_size, n_bins, 2)
                bin_idxs = torch.bucketize(
                    (x - min_bin_edge) / (max_bin_edge - min_bin_edge),
                    self.grid_pts[1:]
                )  # (batch_size,)

                chosen_bins = bins[_batch_idxs, bin_idxs]  # (batch_size, 2)
                min_bin_edge = chosen_bins[:, 0]  # (batch_size,)
                max_bin_edge = chosen_bins[:, 1]  # (batch_size,)

            discretized_normed_lattice_parameters[:, i] = torch.mean(chosen_bins, dim=-1)

        # Check that gamma angles got discretized to valid values
        min_gamma, max_gamma = self.get_valid_gamma_angle_interval(
            raw_alpha_and_beta_angles
        )  # (batch_size,), (batch_size,)
        min_normed_gamma = (self.max_bin_edge - self.min_bin_edge) * (
            (min_gamma - self.MIN_LATTICE_ANGLE) / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
        ) + self.min_bin_edge
        max_normed_gamma = (self.max_bin_edge - self.min_bin_edge) * (
            (max_gamma - self.MIN_LATTICE_ANGLE) / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
        ) + self.min_bin_edge

        gammas_lt_min = (discretized_normed_lattice_parameters[:, -1] < min_normed_gamma)
        gammas_gt_max = (discretized_normed_lattice_parameters[:, -1] > max_normed_gamma)
        if torch.any(gammas_lt_min | gammas_gt_max):
            # -- Enforce lattice gets discretized to something non-singular
            bin_edges = (
                (self.max_bin_edge - self.min_bin_edge) *
                torch.linspace(0, 1, self.n_telescopes * self.n_bins + 1, device=device)
                + self.min_bin_edge
            )  # (n_telescopes * n_bins + 1,)
            bin_midpoints = torch.mean(
                torch.stack([bin_edges[:-1], bin_edges[1:]], dim=-1),
                dim=-1
            )  # (n_telescopes * n_bins,)

            # If less than min_normed_gamma, replace with smallest valid bin
            # (valid means that the bin value is bigger than min_normed_gamma)
            valid_bin_indices = torch.argmax(
                (bin_midpoints[None, :] > min_normed_gamma[:, None])[gammas_lt_min].int(),
                dim=-1
            )  # (num_gammas_lt_min_gamma,)
            discretized_normed_lattice_parameters[:, -1][gammas_lt_min] = (
                bin_midpoints[valid_bin_indices]
            )

            # If greater than max_normed_gamma, replace with largest valid bin
            # (valid means that the bin value is smaller than max_normed_gamma)
            valid_bin_indices = torch.argmax(
                (bin_midpoints[None, :] < max_normed_gamma[:, None])[gammas_gt_max].int(),
                dim=-1
            )  # (num_gammas_gt_max_gamma,)
            discretized_normed_lattice_parameters[:, -1][gammas_gt_max] = (
                bin_midpoints[valid_bin_indices]
            )

        return discretized_normed_lattice_parameters

    @torch.no_grad()
    def get_discretized_raw_lattice_params(
        self, raw_lattice_lengths: Tensor, raw_lattice_angles: Tensor
    ) -> Tuple[Tensor, Tensor]:
        device = self.config.device
        lattice_lengths = raw_lattice_lengths
        lattice_angles = raw_lattice_angles

        # Norm
        normed_lattice_parameters = self._get_normed_lattice_parameters(
            lattice_lengths,
            lattice_angles,
            self.min_bin_edge,
            self.max_bin_edge,
            norm_gamma_separately=False,
        )
        # Discretize
        normed_lattice_parameters = self.get_discretized_normed_lattice_params(
            normed_lattice_parameters, lattice_angles[:, 1:]
        )
        # Denorm
        lattice_params_transform = torch.cat(
            (
                torch.tensor(
                    [
                        self.length_transform(self.MAX_LATTICE_LENGTH)
                        - self.length_transform(self.MIN_LATTICE_LENGTH)
                    ], device=device,
                ).expand(normed_lattice_parameters.shape[0], 3),
                torch.tensor(
                    [self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE], device=device
                ).expand(normed_lattice_parameters.shape[0], 3),
            ),
            dim=1,
        )  # (batch_size, num_lattice_parameters)
        lattice_params_offset = torch.cat(
            (
                torch.tensor(
                    [self.length_transform(self.MIN_LATTICE_LENGTH)],
                    device=device
                ).expand(normed_lattice_parameters.shape[0], 3),
                torch.tensor(
                    [self.MIN_LATTICE_ANGLE], device=device
                ).expand(normed_lattice_parameters.shape[0], 3),
            ),
            dim=1,
        )  # (batch_size, num_lattice_parameters)
        lattice_parameters = (
            (normed_lattice_parameters - self.min_bin_edge)
            / (self.max_bin_edge - self.min_bin_edge)
        ) * lattice_params_transform + lattice_params_offset
        # (batch_size, num_lattice_parameters)
        lattice_lengths = lattice_parameters[:, :3]
        lattice_angles = lattice_parameters[:, 3:]
        return lattice_lengths, lattice_angles