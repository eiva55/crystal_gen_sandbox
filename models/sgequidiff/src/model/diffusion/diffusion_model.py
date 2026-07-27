import cloudpickle
from tqdm import tqdm
import math
import dataclasses

import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap
from torch_scatter import scatter
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

import global_vars
from aliases import *
from constants import MAX_WYCKOFF_SITES
from utils.crystal_utils import uniformly_sample_point_in_asu_wyckoff_site
from utils.io_utils import get_project_dir, DATA_DIRECTORY, SHAPE_DECOMP_DICT_PATH
from utils.wyckoff_shape_decomp_dict_builder import build_wyckoff_shape_decomposition_dict
from model.diffusion.diffusion_utils import (
    get_wyckoff_shape_hull_equations,
    wrap_frac_coords_into_asu,
    d_log_p_asu_wrapped_normal,
    get_space_group_ops_and_conventional_atoms,
    get_wyckoff_projected_gaussian_noise,
)
from model.diffusion.non_equivariant_drift_modules import (
    FourierTimeEmbeddings,
    TorusMLP,
    GNN,
    GNNConfig,
    CSPNet,
    CSPNetConfig,
)

PROJECT_DIR: str = get_project_dir()


@dataclasses.dataclass
class EquivariantDiffusionModelConfig:
    num_wn_lattice_translations: int = 5
    noise_scheduler_num_monte_carlo_samples: int = 10_000
    device: torch.device = "cpu"
    num_timesteps: int = 1000
    sigma_min: float = 0.002
    sigma_max: float = 0.5
    time_emb_dim: int = 256
    model_type: str = "mlp"  # ["mlp", "gnn"]
    num_plane_wave_freqs: int = 64
    subsample_group_operations: bool = False

    # GNN
    gnn_config: GNNConfig = None  # required if model_type == "gnn"
    cspnet_config: CSPNetConfig = None  # required if model_type == "cspnet"


class EquivariantDiffusionModel(nn.Module):
    def __init__(self, config: EquivariantDiffusionModelConfig, device: torch.device = "cpu"):
        self.validate_config(config)
        super().__init__()
        self.config = config
        self.device = config.device
        self.num_wn_lattice_translations = config.num_wn_lattice_translations
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            num_lattice_translations=config.num_wn_lattice_translations,
            num_monte_carlo_samples=config.noise_scheduler_num_monte_carlo_samples,
            device=device,
        )
        self.time_embedder = FourierTimeEmbeddings(dim=config.time_emb_dim)
        self.non_equivariant_drift_model = self.get_non_equivariant_drift_module(
            config, self.time_embedder
        )

        # - Get in/outside tests with hard boundaries in the ASU
        (
            padded_hull_equations,
            # (n_space_groups, n_wyckoffs, max_shapes, n_linear_shape_bounds, 4)
            mask_padded_hull_equations,
            # (n_space_groups, n_wyckoffs, max_shapes)
        ) = get_wyckoff_shape_hull_equations(config.device)
        self.register_buffer(
            "padded_hull_equations", padded_hull_equations
        )
        self.register_buffer(
            "padded_hull_equations_mask", mask_padded_hull_equations
        )

        build_wyckoff_shape_decomposition_dict()
        with open(SHAPE_DECOMP_DICT_PATH.as_posix(), "rb") as f:
            self.wyckoff_shape_decomposition_dict = cloudpickle.load(f)

    def compute_loss(
        self,
        asu_frac_coords: Tensor,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        space_group_indices: Tensor,
        n_atoms_per_xtal: Tensor,
        wyckoff_shape_indices: Tensor,
        lattice_matrices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
    ) -> Tensor:
        """
        Args:
            asu_frac_coords: torch.float
                shape (n_asu_atoms, 3)
            element_indices: torch.long
                shape (n_asu_atoms,)
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            space_group_indices: torch.long
                shape (n_crystals,)
            n_atoms_per_xtal: torch.long
                shape (n_crystals,)
            wyckoff_shape_indices: torch.long
                shape (n_asu_atoms,)
            lattice_matrices: torch.float
                shape (n_crystals, 3, 3)
            lattice_lengths: torch.float
                shape (n_crystals, 3)
            lattice_angles: torch.float
                shape (n_crystals, 3)

        Returns:
            (,) loss
        """
        device = asu_frac_coords.device

        # ------------ Sample 1 noise level per atom
        sampled_timesteps = self.noise_scheduler.uniform_sample_timestep(
            batch_size=space_group_indices.shape[0], device=device
        )
        # (n_crystals,)
        time_embeddings = self.time_embedder(sampled_timesteps.float())
        # (n_crystals, time_emb_dim)

        sampled_timesteps = sampled_timesteps.repeat_interleave(n_atoms_per_xtal, dim=0)
        # (n_asu_atoms,)
        time_embeddings = time_embeddings.repeat_interleave(n_atoms_per_xtal, dim=0)
        # (n_asu_atoms, time_emb_dim)
        sigmas = self.noise_scheduler.sigmas[sampled_timesteps]
        # (n_asu_atoms,)

        with torch.no_grad():
            # ------------ Get noisy frac coords
            # - Add projected Gaussian noise in fractional space
            projected_noise = get_wyckoff_projected_gaussian_noise(
                space_group_indices,
                wyckoff_indices,
                wyckoff_shape_indices,
                n_atoms_per_xtal,
                sigmas[:, None],
            )
            # (n_asu_atoms, 3)
            noisy_asu_frac_coords = asu_frac_coords + projected_noise

            # - Wrap noised coords back into the canonical ASU so we can compute
            # neighboring ASU tiles properly
            _space_group_indices_per_asu_atom = (
                space_group_indices.repeat_interleave(n_atoms_per_xtal, dim=0)
            )
            # (n_asu_atoms,)
            (
                noisy_asu_frac_coords,  # (n_asu_atoms, 3)
                wyckoff_shape_indices,  # (n_asu_atoms,)
            ) = wrap_frac_coords_into_asu(
                noisy_asu_frac_coords,  # (n_asu_atoms, 3)
                wyckoff_indices,        # (n_asu_atoms,)
                space_group_indices,    # (n_crystals,)
                n_atoms_per_xtal,       # (n_crystals,)
                self.padded_hull_equations[
                    _space_group_indices_per_asu_atom, wyckoff_indices
                ],  # (n_asu_atoms, max_shapes, n_linear_bounds, 4)
                self.padded_hull_equations_mask[
                    _space_group_indices_per_asu_atom, wyckoff_indices
                ],  # (n_asu_atoms, max_shapes)
            )
            # ------------ Orbit ground truth frac coords
            (
                _,                              # (n_general_wyckoff_ops, 3, 3)
                _,
                _,
                map_conventional_to_asu_atom,   # (n_general_wyckoff_ops,). Ascending.
                conventional_wyckoff_indices,   # (n_unique_conventional_atoms,)
                conventional_element_indices,   # (n_unique_conventional_atoms,)
                conventional_frac_coords,       # (n_unique_conventional_atoms, 3)
                unique_non_overlapping_atom_indices,  # (n_unique_conventional_atoms,)
            ) = get_space_group_ops_and_conventional_atoms(
                asu_frac_coords,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
            )
            map_unique_conventional_to_asu_atom = (
                map_conventional_to_asu_atom[unique_non_overlapping_atom_indices]
            )

            # ------------ Compute ground truth scores
            ground_truth_scores = d_log_p_asu_wrapped_normal(
                noisy_asu_frac_coords,
                conventional_frac_coords,
                map_unique_conventional_to_asu_atom,
                self.num_wn_lattice_translations,
                sigmas,
            )
            # (n_asu_atoms, 3)

        # ------------ Compute predicted scores and loss
        predicted_scores = self.predict_equivariant_vectors(
            time_embeddings,        # (n_asu_atoms, time_emb_dim)
            noisy_asu_frac_coords,  # (n_asu_atoms, 3)
            element_indices,        # (n_asu_atoms,)
            wyckoff_indices,        # (n_asu_atoms,)
            space_group_indices,    # (n_crystals,)
            n_atoms_per_xtal,       # (n_crystals,)
            lattice_matrices=lattice_matrices,
            lattice_lengths=lattice_lengths,
            lattice_angles=lattice_angles,
        )
        # (n_asu_atoms, 3)

        # - Rescale predicted scores for stability.
        score_norms = self.noise_scheduler.sigma_norms[
            _space_group_indices_per_asu_atom, wyckoff_indices, sampled_timesteps
        ]
        # (n_asu_atoms,)
        loss = F.mse_loss(
            predicted_scores, ground_truth_scores / (score_norms[:, None] + 1e-8)
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        wyckoff_indices: Tensor,
        element_indices: Tensor,
        space_group_indices: Tensor,
        n_atoms_per_xtal: Tensor,
        lattice_matrices: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
        snr: float = 0.4,
        max_step_size: float = 1e6,
    ):
        """
        Variance-exploding Predictor-Corrector SDE sampling of fractional
        coordinates. See Algorithm 2 in https://arxiv.org/pdf/2011.13456.

        Args:
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            element_indices: torch.long
                shape (n_asu_atoms,)
            space_group_indices: torch.long
                shape (n_crystals,)
            n_atoms_per_xtal: torch.long
                shape (n_crystals,)
            lattice_matrices: torch.float
                shape (n_crystals, 3, 3)
            snr: float
                "Signal-to-noise" ratio. See Appendix G in
                https://arxiv.org/pdf/2011.13456.

        Returns:
            trajectory: torch.float
                shape (num_timesteps, n_asu_atoms, 3)
        """
        num_asu_atoms: int = wyckoff_indices.shape[0]
        num_crystals: int = space_group_indices.shape[0]
        device = wyckoff_indices.device
        time_start: int = self.noise_scheduler.num_timesteps
        trajectory: Tensor = torch.zeros(
            self.noise_scheduler.num_timesteps, num_asu_atoms, 3, device=device
        )
        space_group_indices_per_asu_atom = torch.repeat_interleave(
            space_group_indices, n_atoms_per_xtal, dim=0
        )
        # (n_asu_atoms,)
        map_atom_to_xtal = torch.repeat_interleave(
            torch.arange(num_crystals, device=device),
            n_atoms_per_xtal,
            dim=0,
        )  # (n_asu_atoms,)

        # -- Sample from uniform prior in ASU
        (
            x_T,                    # (n_asu_atoms, n_samples_per_wyckoff=1, 3)
            wyckoff_shape_indices,  # (n_asu_atoms, n_samples_per_wyckoff=1)
        ) = uniformly_sample_point_in_asu_wyckoff_site(  # TODO: accelerate this function
            space_group_numbers=[
                str(1 + sg_idx) for sg_idx in space_group_indices_per_asu_atom.tolist()
            ],
            wyckoff_letters=[
                chr(97 + index) if index <= 25 else chr(39 + index)
                for index in wyckoff_indices
            ],
            dictionary_of_wyckoffs_in_asu=global_vars.asu_wyckoff_dict,
            dictionary_of_wyckoff_shape_decompositions=self.wyckoff_shape_decomposition_dict,
            device=device,
            n_samples_per_wyckoff=1,
            return_sampled_wyckoff_shape_indices=True,
        )
        x_T = x_T.squeeze(dim=1)
        # (n_asu_atoms, 3)
        wyckoff_shape_indices = wyckoff_shape_indices.squeeze(dim=1)
        # (n_asu_atoms,)
        space_group_indices_repeated = torch.repeat_interleave(
            space_group_indices, n_atoms_per_xtal, dim=0,
        )
        wyckoff_dims = global_vars.wyckoff_dimension_tensor.to(device)[
            space_group_indices_repeated, wyckoff_indices
        ]
        noise_projection_matrices = global_vars.noise_projection_matrices.to(device)[
            space_group_indices_repeated,
            wyckoff_indices,
            wyckoff_shape_indices,
        ]
        # ^resolve numerical errors that cause the score to point away from the
        # intended Wyckoff positions

        trajectory[time_start-1] = x_T
        for t in tqdm(range(time_start-1, 0, -1)):  # T-1 to 1
            time_embeddings = self.time_embedder(
                torch.tensor([t], dtype=torch.float, device=device)
            ).expand(num_asu_atoms, -1)
            # (n_asu_atoms, time_emb_dim)

            x_t_plus_1 = trajectory[t]
            # (n_asu_atoms, 3)
            sigma_norm_t_plus_1 = self.noise_scheduler.sigma_norms[
                space_group_indices_per_asu_atom, wyckoff_indices, t+1
            ]
            # (n_asu_atoms,)
            sigma_norm_t = self.noise_scheduler.sigma_norms[
                space_group_indices_per_asu_atom, wyckoff_indices, t
            ]
            # (n_asu_atoms,)

            # --- Predictor
            # x_t <- x_t+1 + (sigma_t+1 ** 2 - sigma_t ** 2) * score(x_t+1, sigma_t+1)
            # z ~ N(0, I)
            # x_t <- x_t + sqrt(sigma_t+1 ** 2 - sigma_t ** 2) * z

            # Get score(x_t+1, sigma_t+1)
            predicted_score = sigma_norm_t_plus_1[:, None] * self.predict_equivariant_vectors(
                time_embeddings,
                x_t_plus_1,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
            )  # (n_asu_atoms, 3)
            # Remove numerical errors
            predicted_score = torch.bmm(
                predicted_score[:, None, :], noise_projection_matrices
            ).view(-1, 3)

            sigma_sq_diff = (
                self.noise_scheduler.sigmas[t+1] ** 2
                - self.noise_scheduler.sigmas[t] ** 2
            )  # (,)
            noise = get_wyckoff_projected_gaussian_noise(
                space_group_indices,
                wyckoff_indices,
                wyckoff_shape_indices,
                n_atoms_per_xtal,
                1.0,
            )  # (n_asu_atoms, 3)
            x_t = x_t_plus_1 + sigma_sq_diff * predicted_score
            x_t = x_t + torch.sqrt(sigma_sq_diff) * noise

            # --- Corrector: Algo 4 in https://arxiv.org/pdf/2011.13456
            # z ~ N(0, I)
            # x_t <- x_t + eps_t * score(x_t, sigma_t) + sqrt(2 * eps_t) * z

            # Get score(x_t, sigma_t)
            predicted_score = sigma_norm_t[:, None] * self.predict_equivariant_vectors(
                time_embeddings,
                x_t,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
            )  # (n_asu_atoms, 3)
            # Remove numerical errors
            predicted_score = torch.bmm(
                predicted_score[:, None, :], noise_projection_matrices
            ).view(-1, 3)

            noise = get_wyckoff_projected_gaussian_noise(
                space_group_indices,
                wyckoff_indices,
                wyckoff_shape_indices,
                n_atoms_per_xtal,
                1.0,
            )  # (n_asu_atoms, 3)
            """
            MatterGen version does the below, but also trains with atomic
            density-dependent noise levels [sigma_t(n) = sigma_t / cube_rt(n)].
            
            noise_norm = torch.scatter_add(
                input=torch.zeros_like(space_group_indices, dtype=noise.dtype),
                # (n_crystals,)
                dim=0,
                index=map_atom_to_xtal,
                src=(noise ** 2).sum(dim=-1),  # (n_asu_atoms,)
            ).sqrt().mean()
            grad_norm = torch.scatter_add(
                input=torch.zeros_like(space_group_indices, dtype=noise.dtype),
                dim=0,
                index=map_atom_to_xtal,
                src=(predicted_score ** 2).sum(dim=-1),  # (n_asu_atoms,)
            ).sqrt().mean()
            """
            noise_norm = (noise ** 2).sum(dim=-1).sqrt().mean()  # avg over all atoms in batch
            grad_norm = (predicted_score ** 2).sum(dim=-1).sqrt().mean()
            step_size = 2 * (snr * noise_norm / grad_norm).pow(2)

            # Step size is large when noise is large, which can cause atoms to
            # move out of their Wyckoff positions due to numerical precision
            # issues. To handle this, mask step size with Wyckoff constraints.
            # This can technically happen with sigma_sq_diff in the Predictor
            # step as well, but not in practice for typical values of sigma.
            step_size = torch.where(noise == 0.0, 0.0, step_size)
            # (n_asu_atoms, 3)
            step_size.nan_to_num_(nan=0., posinf=max_step_size, neginf=-max_step_size)

            x_t = x_t + step_size * predicted_score + torch.sqrt(2 * step_size) * noise

            # Project x_t onto Wyckoff shape to correct numerical errors
            x_t = self.project_point_onto_wyckoff_shape(
                x_t,
                space_group_indices_repeated,
                wyckoff_indices,
                wyckoff_shape_indices,
                wyckoff_dims,
            )

            trajectory[t-1] = x_t
        return trajectory % 1.0

    def forward(self, t: Tensor, states: Tuple[Tensor, Tensor]):
        """
        Compute
            velocity = -0.5 * g(t)^2 * score(x, t)
            divergence = -0.5 * g(t)^2 * div_x(score(x, t))
        """
        frac_coords = states[0]
        # (n_asu_atoms, 3)
        n_asu_atoms = frac_coords.shape[0]

        # Interpolate sigma_norm_t at general t
        sigma_norm_t = self.noise_scheduler.get_interpolated_sigma_norm_t(
            t,
            _space_group_indices.repeat_interleave(_n_atoms_per_xtal, dim=0),
            _wyckoff_indices,
        )[:, None]  # (n_asu_atoms, 1)

        with torch.enable_grad():
            frac_coords = frac_coords.detach().requires_grad_()

            time_embeddings = self.time_embedder(t.view(1)).detach().expand(n_asu_atoms, -1)
            # (n_asu_atoms, time_emb_dim)
            dx_dt = (
                0.5 * self.noise_scheduler.d_sigma_sq_dt(float(t))
                * sigma_norm_t * self.predict_equivariant_vectors(
                    time_embeddings,
                    frac_coords,
                    _element_indices,
                    _wyckoff_indices,
                    _space_group_indices,
                    _n_atoms_per_xtal,
                    _lattice_matrices,
                    _lattice_lengths,
                    _lattice_angles,
                    _differentiate_graph_construction,
                )  # score
            )  # (n_asu_atoms, 3)

            # Project dx_dt onto Wyckoff shape to protect against potential
            # numerical precision issues causing atoms to move out of their
            # Wyckoff positions. Might not be necessary but doesn't really add
            # much compute.
            dx_dt = torch.bmm(dx_dt[:, None, :], _noise_projection_matrices).view(-1, 3)

            divergence = _divergence_fn(dx_dt, frac_coords)

        frac_coords.grad = None
        dx_dt.grad = None
        return dx_dt.detach(), divergence

    @torch.no_grad()
    def get_log_likelihood(
        self,
        frac_coords: Tensor,            # (n_asu_atoms, 3)
        element_indices: Tensor,        # (n_asu_atoms,)
        wyckoff_indices: Tensor,        # (n_asu_atoms,)
        space_group_indices: Tensor,    # (n_crystals,)
        n_atoms_per_xtal: Tensor,       # (n_crystals,)
        lattice_matrices: Tensor,       # (n_crystals, 3, 3)
        lattice_lengths: Tensor,        # (n_crystals, 3)
        lattice_angles: Tensor,         # (n_crystals, 3)
        divergence_type: str = "full",
        num_ode_solver_steps: int = 500,
    ):
        assert divergence_type in ["full", "hutchinson"]
        device = frac_coords.device

        # identify linear Wyckoff subspaces that frac_coords reside in
        space_group_indices_per_asu_atom = torch.repeat_interleave(
            space_group_indices, n_atoms_per_xtal, dim=0
        )  # (n_asu_atoms,)
        (
            frac_coords,            # (n_asu_atoms, 3)
            wyckoff_shape_indices,  # (n_asu_atoms,)
        ) = wrap_frac_coords_into_asu(
            frac_coords,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
            self.padded_hull_equations[
                space_group_indices_per_asu_atom, wyckoff_indices
            ],  # (n_asu_atoms, max_shapes, n_linear_bounds, 4)
            self.padded_hull_equations_mask[
                space_group_indices_per_asu_atom, wyckoff_indices
            ],  # (n_asu_atoms, max_shapes)
        )

        # integration times
        T = float(self.noise_scheduler.num_timesteps)  # noise distribution
        t0 = 0.0  # data distribution

        # init stuff
        x0_samples = frac_coords
        logp_dummy = torch.zeros(frac_coords.shape[0], device=device)
        global _element_indices
        global _wyckoff_indices
        global _space_group_indices
        global _n_atoms_per_xtal
        global _divergence_fn
        global _noise_projection_matrices
        global _lattice_matrices
        global _lattice_lengths
        global _lattice_angles
        global _differentiate_graph_construction
        _element_indices = element_indices
        _wyckoff_indices = wyckoff_indices
        _space_group_indices = space_group_indices
        _n_atoms_per_xtal = n_atoms_per_xtal
        _lattice_matrices = lattice_matrices
        _lattice_lengths = lattice_lengths
        _lattice_angles = lattice_angles
        _differentiate_graph_construction = True
        _divergence_fn = (
            divergence_full if divergence_type == "full" else divergence_hutch
        )
        _noise_projection_matrices = global_vars.noise_projection_matrices.to(device)[
            space_group_indices_per_asu_atom,
            wyckoff_indices,
            wyckoff_shape_indices,
        ]  # (n_asu_atoms, 3, 3)

        # integrate
        xt_samples, logp_diff_t = odeint(
            self,
            (x0_samples, logp_dummy),
            torch.tensor([t0, T], device=device, dtype=torch.float),
            method='rk4',  # 'dopri5',
            options=dict(step_size=T/num_ode_solver_steps),
            #atol=1e-5,  # atol and rtol are for adaptive solvers like dopri5
            #rtol=1e-5,
        )  # (n_integration_times, n_samples, D=3)
        # xT_samples = xt_samples[-1]  # from uniform prior
        logp_diff = logp_diff_t[-1]  # integral from 0 to T

        logp_T = self.uniform_prior_log_prob(
            wyckoff_indices, space_group_indices, n_atoms_per_xtal
        )  # (n_asu_atoms,)
        logp = logp_diff + logp_T
        return logp  # (n_asu_atoms,)

    def predict_equivariant_vectors(
        self,
        time_embeddings: Tensor,
        frac_coords: Tensor,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        space_group_indices: Tensor,
        n_atoms_per_xtal: Tensor,
        lattice_matrices: Tensor = None,
        lattice_lengths: Tensor = None,
        lattice_angles: Tensor= None,
        differentiate_graph_construction: bool = False,
    ):
        """
        Symmetrization as mean(A^-1 @ f(Ax + t))

        Args:
            time_embeddings: torch.float
                (n_asu_atoms, time_emb_dim)
            frac_coords: torch.float
                (n_asu_atoms, 3)
            element_indices: torch.long
                (n_asu_atoms,)
            wyckoff_indices: torch.long
                (n_asu_atoms,)
            space_group_indices: torch.long
                (n_crystals,)
            n_atoms_per_xtal: torch.long
                (n_crystals,)
            lattice_matrices: torch.float
                (n_crystals, 3, 3)
            lattice_lengths: torch.float
                (n_crystals, 3)
            lattice_angles: torch.float
                (n_crystals, 3)
        """
        device = frac_coords.device
        frac_coords = frac_coords % 1.0  # translation does not affect direction vectors

        (
            A_inv_ops,                      # (n_general_wyckoff_ops, 3, 3)
            t_ops,                          # (n_general_wyckoff_ops, 1, 3)
            inverse_indices,                # (n_general_wyckoff_ops,)
            map_conventional_to_asu_atom,   # (n_general_wyckoff_ops,). Ascending.
            conventional_wyckoff_indices,   # (n_unique_conventional_atoms,)
            conventional_element_indices,   # (n_unique_conventional_atoms,)
            frac_coords_of_conv_atoms,      # (n_unique_conventional_atoms, 3)
            unique_non_overlapping_atom_indices,  # (n_unique_conventional_atoms,)
        ) = get_space_group_ops_and_conventional_atoms(
            frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
        )

        if self.training and self.config.subsample_group_operations:
            # todo: not sure this really makes sense... subsampling means
            #   removing nodes from the conventional cell graph
            # todo: need to make sure the ground truth score we match to is for
            #   the subsampled atom. Requires computing the Wyckoff shape index
            #   of the subsampled atom, which may not be in the ASU. We do not
            #   have the in/outside test matrices for those shapes currently
            #   pre-computed.
            raise NotImplementedError
            # -- Uniformly sub-sample 1 group operation per atom

            # - Identify first occurrence of each unique value
            first_unique_mask = torch.cat(
                [
                    torch.tensor([True], device=device),
                    map_conventional_to_asu_atom[1:] != map_conventional_to_asu_atom[:-1]
                ]
            )  # (n_general_wyckoff_ops,)

            # - Shuffle within each atom orbit to get a random group op per atom
            randperm = torch.randperm(n=map_conventional_to_asu_atom.shape[0], device=device)
            # (n_general_wyckoff_ops,)
            selected_indices = torch.argmax(
                randperm *
                (map_conventional_to_asu_atom == map_conventional_to_asu_atom.unsqueeze(1)),
                dim=1
            )[first_unique_mask]
            # (n_asu_atoms,)

            # - Select 1 group op per atom
            conventional_element_indices = (
                conventional_element_indices[inverse_indices][selected_indices]
            )  # (n_asu_atoms,)
            frac_coords_of_conv_atoms = (
                frac_coords_of_conv_atoms[inverse_indices][selected_indices]
            )  # (n_asu_atoms, 3)
            n_conv_atoms_per_xtal = n_atoms_per_xtal
            # (n_crystals,)
            vector_field = self.non_equivariant_drift_model(
                frac_coords=frac_coords_of_conv_atoms,
                element_indices=conventional_element_indices,
                n_atoms_per_xtal=n_conv_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
                time_embeddings=time_embeddings,
                differentiate_graph_construction=differentiate_graph_construction,
            )  # (n_asu_atoms, 3)
        else:
            map_asu_atom_to_xtal = torch.repeat_interleave(
                torch.arange(space_group_indices.shape[0], device=device),
                n_atoms_per_xtal,
                dim=0
            )  # (n_crystals,)
            map_unique_conventional_to_asu_atom = map_conventional_to_asu_atom[
                unique_non_overlapping_atom_indices
            ]  # (n_unique_conventional_atoms,)
            n_conv_atoms_per_xtal = torch.scatter_add(
                input=torch.zeros_like(n_atoms_per_xtal),
                dim=0,
                index=map_asu_atom_to_xtal[map_unique_conventional_to_asu_atom],
                src=torch.ones_like(conventional_wyckoff_indices),
            )  # (n_crystals,)

            # Scatter mean
            src = torch.bmm(
                self.non_equivariant_drift_model(
                    frac_coords=frac_coords_of_conv_atoms,
                    element_indices=conventional_element_indices,
                    n_atoms_per_xtal=n_conv_atoms_per_xtal,
                    lattice_matrices=lattice_matrices,
                    lattice_lengths=lattice_lengths,
                    lattice_angles=lattice_angles,
                    time_embeddings=time_embeddings[map_unique_conventional_to_asu_atom],
                    differentiate_graph_construction=differentiate_graph_construction,
                )[inverse_indices].view(-1, 1, 3),
                A_inv_ops,
            ).squeeze(dim=1)
            # (n_general_wyckoff_ops, 3)
            vector_field = scatter(
                src=src,
                index=map_conventional_to_asu_atom,
                dim=0,
                dim_size=frac_coords.shape[0],
                reduce='mean',
            )  # (n_asu_atoms, 3)
        return vector_field

    @torch.no_grad()
    def uniform_prior_log_prob(
        self,
        wyckoff_indices: Tensor,
        space_group_indices: Tensor,
        n_atoms_per_xtal: Tensor,
    ):
        """
        Args:
            wyckoff_indices: torch.long
                shape (n_asu_atoms,)
            space_group_indices: torch.long
                shape (n_crystals,)
            n_atoms_per_xtal: torch.long
                shape (n_crystals,)

        Returns:
            shape (n_asu_atoms,)
        """
        device = wyckoff_indices.device
        space_group_indices_per_atom = torch.repeat_interleave(
            space_group_indices, n_atoms_per_xtal, dim=0
        )  # (n_asu_atoms,)
        wyckoff_volumes = torch.sum(
            global_vars.wyckoff_shape_volumes.to(device)[
                space_group_indices_per_atom, wyckoff_indices
            ],  # (n_asu_atoms, max_wyckoff_shapes)
            dim=-1
        )  # (n_asu_atoms,)
        return torch.where(
            wyckoff_volumes == 0.0,  # 0D Wyckoff positions have probability 1
            0.0,
            torch.log(1.0 / wyckoff_volumes),
        )

    @torch.no_grad()
    def wrap_trajectory_into_asu(self):
        """For visualization purposes"""
        raise NotImplementedError

    @staticmethod
    def validate_config(config: EquivariantDiffusionModelConfig):
        assert (
               config.sigma_min > 0.0 and
               isinstance(config.time_emb_dim, int) and
               config.time_emb_dim > 0 and
               config.model_type in ["mlp", "gnn", "cspnet"]
        )

    @staticmethod
    def get_non_equivariant_drift_module(
        config: EquivariantDiffusionModelConfig,
        time_embedder: FourierTimeEmbeddings,
    ) -> nn.Module:
        """
        Returned model has forward function which outputs non-equivariant
        (n_atoms, 3) vectors
        """
        if config.model_type == "mlp":
            non_equivariant_drift_model = TorusMLP(
                time_embedder,
                config.num_plane_wave_freqs,
            )
        elif config.model_type == "gnn":
            non_equivariant_drift_model = GNN(config.gnn_config, time_embedder)
        elif config.model_type == "cspnet":
            non_equivariant_drift_model = CSPNet(config.cspnet_config, time_embedder)
        else:
            raise AttributeError
        return non_equivariant_drift_model

    @staticmethod
    def project_point_onto_wyckoff_shape(
        points: Tensor,
        space_group_indices_repeated: Tensor,
        wyckoff_indices: Tensor,
        wyckoff_shape_indices: Tensor,
        wyckoff_dims: Tensor,
    ) -> Tensor:
        def project_points_to_lines(points, p0s, dirs):
            # Normalize direction vectors
            dirs = dirs / dirs.norm(dim=-1, keepdim=True)  # (B, 3)

            # Vector from line points to the given points
            v = points - p0s  # (B, 3)

            # Projection scalar
            t = (v * dirs).sum(dim=-1, keepdim=True)  # (B, 1)

            # Projected point on line
            proj = p0s + t * dirs  # (B, 3)
            return proj

        def project_points_to_planes(points, p0s, normals):
            # Normalize normal vectors
            normals = normals / normals.norm(dim=-1, keepdim=True)  # (B, 3)

            # Vector from plane points to the given points
            v = points - p0s  # (B, 3)

            # Distance from point to plane along the normal
            dist = (v * normals).sum(dim=-1, keepdim=True)  # (B, 1)

            # Projected point on plane
            proj = points - dist * normals  # (B, 3)
            return proj

        device = points.device
        wyckoff_dim_is_1 = wyckoff_dims == 1
        wyckoff_dim_is_2 = wyckoff_dims == 2
        if torch.any(wyckoff_dim_is_1):
            points[wyckoff_dim_is_1] = project_points_to_lines(
                points=points[wyckoff_dim_is_1],
                p0s=global_vars.point_per_1d_wyckoff_line.to(device)[
                    space_group_indices_repeated[wyckoff_dim_is_1],
                    wyckoff_indices[wyckoff_dim_is_1],
                    wyckoff_shape_indices[wyckoff_dim_is_1],
                ],
                dirs=global_vars.line_directions_of_1d_wyckoffs.to(device)[
                    space_group_indices_repeated[wyckoff_dim_is_1],
                    wyckoff_indices[wyckoff_dim_is_1],
                    wyckoff_shape_indices[wyckoff_dim_is_1],
                ],
            )
        if torch.any(wyckoff_dim_is_2):
            points[wyckoff_dim_is_2] = project_points_to_planes(
                points=points[wyckoff_dim_is_2],
                p0s=global_vars.point_per_2d_wyckoff_plane.to(device)[
                    space_group_indices_repeated[wyckoff_dim_is_2],
                    wyckoff_indices[wyckoff_dim_is_2],
                    wyckoff_shape_indices[wyckoff_dim_is_2],
                ],
                normals=global_vars.plane_normals_of_2d_wyckoffs.to(device)[
                    space_group_indices_repeated[wyckoff_dim_is_2],
                    wyckoff_indices[wyckoff_dim_is_2],
                    wyckoff_shape_indices[wyckoff_dim_is_2],
                ]
            )
        return points


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        sigma_norm_type: str = "asu_wrapped",
        num_lattice_translations: int = 5,
        num_monte_carlo_samples: int = 10_000,
        device: torch.device = "cpu",
    ):
        """
        Exponential scheduler:
            sigma_0 = 0                                             if t = 0
            sigma_t = sigma_1 * (sigma_T / sigma_1)^(t-1 / T-1)     if t > 0
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_lattice_translations = num_lattice_translations
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_timesteps)),
            dtype=torch.float, device=device
        )
        # (num_timesteps,)

        if sigma_norm_type == "unwrapped":
            _sigma_norms = self._sigma_norm_unwrapped(sigmas)
        elif sigma_norm_type == "asu_wrapped":
            _sigma_norms = self._sigma_norm_asu_wrapped(sigmas, num_monte_carlo_samples)
        else:
            raise AttributeError
        # (230, MAX_WYCKOFF_SITES, num_timesteps)

        self.register_buffer(
            "sigmas", torch.cat([torch.zeros([1], device=sigmas.device), sigmas], dim=0)
        )  # (1 + num_timesteps,)
        self.register_buffer(
            "sigma_norms",
            torch.cat(
                [torch.ones(230, MAX_WYCKOFF_SITES, 1, device=_sigma_norms.device), _sigma_norms],
                dim=-1
            )
        )  # (230, MAX_WYCKOFF_SITES, 1 + num_timesteps)

    def uniform_sample_timestep(self, batch_size: int, device: torch.device):
        return torch.randint(
            low=1,
            high=self.num_timesteps+1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def d_sigma_sq_dt(self, t: float) -> float:
        """
        Useful for ODE sampling since g(t)=sqrt(d_sigma^2 / dt).

        Exponential scheduler:
            sigma_0 = 0                                             if t = 0
            sigma_t = sigma_1 * (sigma_T / sigma_1)^(t-1 / T-1)     if t > 0
        
        d_sigma^2/dt = 2 * d_sigma/dt
                     = 2 * sigma_1 * log(sigma_T / sigma_1) * (sigma_T / sigma_1)^((t-1)/(T-1))
        """
        t = max(t, 1e-5)

        T = float(self.num_timesteps)
        sigma_ratio = self.sigma_max / self.sigma_min
        d_sigma_sq_dt = 2 * (
            (self.sigma_min / (T-1))
            * math.log(sigma_ratio)
            * (sigma_ratio ** ((t-1) / (T-1)))
        )
        return d_sigma_sq_dt

    @torch.no_grad()
    def _sigma_norm_unwrapped(self, sigmas: Tensor):
        """
        Returns 1/sigma as in arXiv:1907.05600
        """
        sigma_norms = 1.0 / sigmas
        # (num_timesteps,)
        return sigma_norms[None, None, :].expand(230, MAX_WYCKOFF_SITES, -1)

    @torch.no_grad()
    def _sigma_norm_asu_wrapped(self, sigmas: Tensor, num_monte_carlo_samples: int = 10_000):
        """
        Compute expected 2-norms of scores wrapped with space group ops. Returns
        them in a padded tensor.
            \mathbb{E}_{xt ~ p(xt|x0), x_0 ~ Unif} [||score||^2_2]

        Args:
            sigmas: torch.float
                shape (num_timesteps,)

        Returns:
            shape (230, MAX_WYCKOFF_SITES, num_timesteps)
        """
        num_timesteps = sigmas.shape[0]
        num_lattice_translations: int = self.num_lattice_translations
        device = sigmas.device
        filename: str = (
            DATA_DIRECTORY / f"expected_score_norms_minSigma{sigmas[0]:0.3f}_maxSigma{sigmas[-1]:0.3f}_T{num_timesteps}_{num_monte_carlo_samples}MCsamples_{num_lattice_translations}LatticeTranslations.pt"
        ).as_posix()
        if os.path.exists(filename):
            print(f"Loaded {filename}")
            sigma_norms = torch.load(filename, map_location=device)
            return sigma_norms

        build_wyckoff_shape_decomposition_dict()
        with open(SHAPE_DECOMP_DICT_PATH.as_posix(), "rb") as f:
            wyckoff_shape_decomp_dict = cloudpickle.load(f)

        # Vmap functions
        vmapped_get_wyckoff_projected_gaussian_noise = vmap(
            vmap(
                get_wyckoff_projected_gaussian_noise,
                in_dims=(None, None, 1, None, None),
                out_dims=1,
                randomness="different",
            ),  # vmap over monte carlo samples
            in_dims=(None, None, None, None, 0),
            out_dims=2,
            randomness="different",
        )  # vmap over sigmas
        vmapped_d_log_p_asu_wrapped_normal = vmap(
            d_log_p_asu_wrapped_normal,
            in_dims=(1, None, None, None, 1),
            out_dims=1
        )  # vmap over sigmas

        # ---
        # Sample x0 uniformly, then xt with projected ASU-wrapped normal.
        # ---
        asu_wyckoff_dict: Dict = global_vars.asu_wyckoff_dict
        sigma_norms = torch.zeros(
            230, MAX_WYCKOFF_SITES, num_timesteps, dtype=torch.float, device=device
        )
        for sg_num in tqdm(range(230, 0, -1)):  # do memory-intensive groups first for debugging
            space_group_dict: Dict = asu_wyckoff_dict[str(sg_num)]
            wyckoff_letters: List[str] = space_group_dict["ordered_wyckoff_letters"]

            # Sample points uniformly randomly in each Wyckoff position
            (
                x0s,                    # (n_wyckoffs, num_monte_carlo_samples, 3)
                wyckoff_shape_indices,  # (n_wyckoffs, num_monte_carlo_samples)
            ) = uniformly_sample_point_in_asu_wyckoff_site(
                space_group_numbers=[str(sg_num)] * len(wyckoff_letters),
                wyckoff_letters=wyckoff_letters,
                dictionary_of_wyckoffs_in_asu=asu_wyckoff_dict,
                dictionary_of_wyckoff_shape_decompositions=wyckoff_shape_decomp_dict,
                device=device,
                n_samples_per_wyckoff=num_monte_carlo_samples,
                return_sampled_wyckoff_shape_indices=True,
            )

            space_group_indices = torch.tensor([sg_num-1], dtype=torch.long, device=device)
            wyckoff_indices = torch.arange(len(wyckoff_letters), device=device)
            for i in range(len(wyckoff_letters)):
                # Sample xt for each noise level
                _wyckoff_indices = wyckoff_indices[i][None]
                projected_noise = vmapped_get_wyckoff_projected_gaussian_noise(
                    space_group_indices,
                    _wyckoff_indices,
                    wyckoff_shape_indices[i].view(1, -1),
                    torch.tensor([1], dtype=torch.long, device=device),
                    sigmas,
                )
                # (1, num_monte_carlo_samples, num_timesteps, 3)
                xts = x0s[i, :, None, :] + projected_noise[0]
                # (num_monte_carlo_samples, num_timesteps, 3)

                _wyckoff_indices = _wyckoff_indices.expand(num_monte_carlo_samples)
                # (num_monte_carlo_samples,)
                (
                    _,                      # (n_general_wyckoff_ops, 3, 3)
                    _,                          # (n_general_wyckoff_ops, 1, 3)
                    _,                # (n_general_wyckoff_ops,)
                    map_conventional_to_asu_atom,   # (n_general_wyckoff_ops,). Ascending.
                    conventional_wyckoff_indices,   # (n_unique_conventional_atoms,)
                    _,   # (n_unique_conventional_atoms,)
                    orbited_x,                      # (n_unique_conventional_atoms, 3)
                    unique_non_overlapping_atom_indices,  # (n_unique_conventional_atoms,)
                ) = get_space_group_ops_and_conventional_atoms(
                    x0s[i],  # (num_monte_carlo_samples, 3)
                    torch.zeros_like(_wyckoff_indices),  # (num_mc_samples,)
                    _wyckoff_indices,       # (num_mc_samples,)
                    space_group_indices,  # (1,)
                    n_atoms_per_xtal=torch.tensor([num_monte_carlo_samples], dtype=torch.long, device=device),  # (1,)
                )

                # Compute norms over batches of timesteps to avoid OOMs
                for sigma_idxs in torch.chunk(
                    torch.arange(sigmas.shape[0], device=device), chunks=200
                ):
                    pdf = vmapped_d_log_p_asu_wrapped_normal(
                        xts[:, sigma_idxs],  # (num_monte_carlo_samples, batch_timesteps, 3)
                        orbited_x,  # (num_conventional_atoms, 3)
                        map_conventional_to_asu_atom[unique_non_overlapping_atom_indices],  # (num_conventional_atoms,)
                        num_lattice_translations,  # number of translations
                        sigmas[sigma_idxs][None, :].expand(num_monte_carlo_samples, -1),  # (num_monte_carlo_samples, batch_timesteps)
                    )  # (num_mc_samples, batch_timesteps, 3)
                    sigma_norms[sg_num-1, i, sigma_idxs] = (
                        (pdf ** 2).sum(dim=-1).sqrt().mean(dim=0)
                    )  # (batch_timesteps,)

        torch.save(sigma_norms, filename)
        return sigma_norms

    @torch.no_grad()
    def get_interpolated_sigma_norm_t(
        self,
        t: Tensor,                    # (,)
        space_group_indices: Tensor,  # (n,)
        wyckoff_indices: Tensor,      # (n,)
    ) -> Tensor:
        assert 0.0 <= t <= self.num_timesteps
        t_below = torch.floor(t)
        t_above = torch.ceil(t)
        if t_below == t_above:
            if t_below == 0.0:
                t_below = (0.0 * t_below)        # 0
                t_above = 1.0 + (0.0 * t_below)  # 1
            elif t_below == self.num_timesteps:
                t_above = self.num_timesteps + (0.0 * t_below)      # T-1
                t_below = self.num_timesteps - 1 + (0.0 * t_below)  # T-2
            else:
                t_above = t_below + 1

        p = (t - t_below) / (t_above - t_below)
        sigma_norm_t_below = self.sigma_norms[
            space_group_indices,
            wyckoff_indices,
            t_below.to(dtype=torch.long)[None].expand(space_group_indices.shape[0])
        ]  # (n,)
        sigma_norm_t_above = self.sigma_norms[
            space_group_indices,
            wyckoff_indices,
            t_above.to(dtype=torch.long)[None].expand(space_group_indices.shape[0])
        ]  # (n,)
        return (1 - p) * sigma_norm_t_below + p * sigma_norm_t_above  # (n,)


def divergence_full(x_dot: Tensor, x: Tensor) -> Tensor:
    """
    Calculate the time derivative of the log determinant of the
    Jacobian. This is -Tr(\partial z_dot / \partial z(t)).
    Note that this is the brute force, O(D^2), method. This is fine
    for D=3, but we can use a Monte-carlo estimate of the trace
    (Hutchinson's trace estimator) for a linear time estimate in
    larger dimensions.

    Args:
        x_dot: shape (n, 3)
        x: shape (n, 3)

    Returns:
        shape (n,)
    """
    divergence = torch.zeros(x.shape[0], device=x.device, requires_grad=False)
    for i in range(x.shape[1]):
        divergence.add_(
            torch.autograd.grad(
                x_dot[:, i].sum(), x, create_graph=False, retain_graph=(i < x.shape[1] - 1)
            )[0][:, i]
        )
    divergence = -divergence.view(-1, 1)
    return divergence.view(-1, 1).detach()


def divergence_hutch(x_dot, x):
    """
    Calculate the time derivative of the log determinant of the
    Jacobian. This is -Tr(\partial z_dot / \partial z(t)).
    Here we use a Monte-carlo estimate of the trace (Hutchinson's trace
    estimator) for a linear time estimate in larger dimensions.

    Args:
        x_dot: shape (n, 3)
        x: shape (n, 3)

    Returns:
        shape (n,)
    """
    # Rademacher noise is -1 or +1 w.p. 1/2
    eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1

    fn_eps = torch.sum(x_dot * eps)
    grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    divergence = torch.sum(eps * grad_fn_eps, dim=tuple(range(1, len(x.shape))))
    return -divergence.detach()

