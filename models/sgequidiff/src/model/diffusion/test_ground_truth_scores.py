from aliases import *
from model.diffusion.diffusion_model import (
    EquivariantDiffusionModel,
    EquivariantDiffusionModelConfig,
)
import global_vars

from model.diffusion.diffusion_utils import (
    get_training_crystals,
    d_log_p_asu_wrapped_normal,
    wrap_frac_coords_into_asu,
    get_space_group_ops_and_conventional_atoms,
)


def _test_d_log_p_asu_wrapped_normal(sigma = 1.0):
    config = EquivariantDiffusionModelConfig(
        num_wn_lattice_translations=2,
        noise_scheduler_num_monte_carlo_samples=10,
        time_emb_dim=64,
    )
    model = EquivariantDiffusionModel(config)
    (
        asu_frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        wyckoff_shape_indices,
        lattice_matrices,
        _,
        _,
    ) = get_training_crystals(n=1)
    device = asu_frac_coords.device

    # Get the first atom
    asu_frac_coords = asu_frac_coords[0].view(1, 3)
    element_indices = element_indices[0].view(1)
    wyckoff_indices = wyckoff_indices[0].view(1)
    n_atoms_per_xtal = torch.tensor([1], dtype=torch.long)
    wyckoff_shape_indices = wyckoff_shape_indices[0].view(1)

    # Add projected Gaussian noise in fractional space
    unprojected_fractional_noise = sigma * torch.randn_like(asu_frac_coords)
    # (n_asu_atoms, 3)
    projected_fractional_noise = torch.bmm(
        unprojected_fractional_noise[:, None, :],
        global_vars.noise_projection_matrices.to(device)[
            space_group_indices.repeat_interleave(n_atoms_per_xtal, dim=0),
            wyckoff_indices,
            wyckoff_shape_indices
        ]
    ).view(-1, 3)
    # (n_asu_atoms, 3)
    noisy_asu_frac_coords = asu_frac_coords + projected_fractional_noise

    # Wrap noised coords back into the canonical ASU so we can compute
    # neighboring ASU tiles properly
    _space_group_indices_per_atom = (
        space_group_indices.repeat_interleave(n_atoms_per_xtal, dim=0)
    )
    (
        noisy_asu_frac_coords,  # (n_asu_atoms, 3)
        wyckoff_shape_indices,  # (n_asu_atoms,)
    ) = wrap_frac_coords_into_asu(
        noisy_asu_frac_coords,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        model.padded_hull_equations[
            _space_group_indices_per_atom, wyckoff_indices
        ],
        model.padded_hull_equations_mask[
            _space_group_indices_per_atom, wyckoff_indices
        ],
    )

    with torch.no_grad():
        (
            _,                          # (n_general_wyckoff_ops, 3, 3)
            _,
            _,
            map_conventional_to_asu_atom,   # (n_general_wyckoff_ops,). Ascending.
            _,   # (n_unique_conventional_atoms,)
            _,   # (n_unique_conventional_atoms,)
            conventional_frac_coords,       # (n_unique_conventional_atoms, 3)
            unique_non_overlapping_atom_indices,  # (n_unique_conventional_atoms,)
        ) = get_space_group_ops_and_conventional_atoms(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
        )
        map_conventional_to_asu_atom = map_conventional_to_asu_atom[unique_non_overlapping_atom_indices]

    # Compute our gradient
    our_grad = d_log_p_asu_wrapped_normal(
        noisy_asu_frac_coords,
        conventional_frac_coords,
        map_conventional_to_asu_atom,
        n_lattice_translations=2,
    )

    # Compute GT gradient
    def _p_asu_wrapped_normal(
        noisy_asu_frac_coords,
        asu_frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
    ):
        (
            _,
            _,
            _,
            map_conventional_to_asu_atom,   # (n_general_wyckoff_ops,). Ascending.
            _,
            _,
            conventional_frac_coords,       # (n_unique_conventional_atoms, 3)
            unique_non_overlapping_atom_indices,
        ) = get_space_group_ops_and_conventional_atoms(
            asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
        )

        n_lattice_translations = 2
        device = conventional_frac_coords.device
        translations_1d = torch.arange(-n_lattice_translations, n_lattice_translations+1, device=device)
        # (n_lattice_translations_per_dim,)
        translations = torch.cartesian_prod(translations_1d, translations_1d, translations_1d)
        # (n_lattice_translations, 3)

        # - Filter out conventional atoms that are in a different Wyckoff shape
        # - than noisy_frac_coord
        x = conventional_frac_coords[:, None, :] + translations[None, :, :]
        # (n_conventional_atoms, n_lattice_translations, 3)

        # For k lattice translations per dim, sum over k^3 grid
        noisy_x_minus_gt_xs = (
            noisy_asu_frac_coords[map_conventional_to_asu_atom[unique_non_overlapping_atom_indices]][:, None, :]
            - x
        )
        # (n_conventional_atoms, n_lattice_translations, 3)
        p = torch.sum(
            torch.exp(
                -(noisy_x_minus_gt_xs ** 2).sum(dim=-1) / 2 / sigma ** 2
            ),
            # (n_conventional_atoms, n_lattice_translations)
            dim=1
        )  # (n_conventional_atoms,)

        # Sum over space group ops
        # Cannot call jacfwd on scatter ops. Assume we only have 1 ASU atom.
        p = p.sum()
        return p

    noisy_asu_frac_coords.requires_grad_(True)
    gt_grad = torch.autograd.grad(
        torch.log(
            _p_asu_wrapped_normal(
                noisy_asu_frac_coords,
                asu_frac_coords,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
            )
        ),
        noisy_asu_frac_coords,
    )[0]
    print(our_grad.shape, gt_grad.shape)
    print(our_grad, gt_grad)
    assert torch.all(torch.isclose(our_grad, gt_grad)), f"{our_grad - gt_grad}"


if __name__ == "__main__":
    _test_d_log_p_asu_wrapped_normal()
