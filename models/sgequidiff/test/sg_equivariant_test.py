from tqdm import tqdm

import torch.nn as nn
from torch.distributions import Normal
from torch_scatter import scatter
from pyxtal.symmetry import Group as PyxtalGroup

from custom_dataset import AsymmetricUnitDataset
import global_vars
from aliases import *
from model.diffusion.diffusion_utils import (
    get_space_group_ops_and_conventional_atoms
)


class SpaceGroupEquivariantVectorField(nn.Module):
    def __init__(
            self,
            num_plane_wave_freqs: int = 64,
            max_plane_wave_freq: int = 512,
            plane_wave_fourier_scale: float = 1.0,
    ):
        super().__init__()
        self.subsample_group_operations = False
        self.isotropic_plane_waves = False
        self.fourier_scale = plane_wave_fourier_scale

        # Featurize fractional coordinates with plane waves
        self.max_freq = max_plane_wave_freq
        self.num_freqs = num_plane_wave_freqs
        plane_wave_freqs = self._get_plane_wave_frequencies(
            self.num_freqs, self.max_freq, self.fourier_scale, self.isotropic_plane_waves
        )  # (3, num_freqs)
        self.register_buffer("plane_wave_freqs", plane_wave_freqs)
        # (3, num_freqs)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.num_freqs, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
        self.get_space_group_ops_and_conventional_atoms: Callable = (
            get_space_group_ops_and_conventional_atoms
        )

    @staticmethod
    def _get_plane_wave_frequencies(
        num_freqs: int,
        max_freq: int = 512,
        fourier_scale: float = 1.0,
        isotropic_plane_waves: bool = False
    ):
        if isotropic_plane_waves:
            plane_wave_freqs = torch.linspace(
                1, num_freqs, steps=num_freqs
            ).view(1, -1).expand(3, -1)
            # (3, num_freqs)
        else:  # Sample plane wave frequencies from discretized Gaussian
            freqs_1d_grid = torch.linspace(-max_freq, max_freq, steps=1 + 2 * max_freq)
            freqs_1d_grid = freqs_1d_grid[freqs_1d_grid != 0.0]  # remove trivial frequency
            normal = Normal(loc=torch.tensor([0.0]), scale=fourier_scale)
            probs = normal.log_prob(freqs_1d_grid).exp()
            # Sampling from 3 iid Gaussians saves memory

            plane_wave_freqs = torch.tensor([[], [], []])
            samples_per_iter = 2 * num_freqs
            max_iters = 100
            iteration = 0
            while plane_wave_freqs.shape[-1] < num_freqs or iteration > max_iters:
                # Rejection sample for unique anisotropic plane waves
                iteration += 1
                kx = freqs_1d_grid[
                    torch.multinomial(
                        probs, num_samples=samples_per_iter, replacement=True,
                    )
                ]  # (num_freqs,)
                ky = freqs_1d_grid[
                    torch.multinomial(
                        probs, num_samples=samples_per_iter, replacement=True,
                    )
                ]
                kz = freqs_1d_grid[
                    torch.multinomial(
                        probs, num_samples=samples_per_iter, replacement=True,
                    )
                ]
                plane_wave_freqs = torch.cat(
                    [plane_wave_freqs, torch.stack([kx, ky, kz])], dim=-1
                )
                plane_wave_freqs = torch.unique(plane_wave_freqs, dim=-1)
        return plane_wave_freqs[:, :num_freqs]

    def plane_wave_fourier_features(self, x: Tensor):
        """
        Args:
            x: torch.float
                (n, 3)

        Returns:
            (n, 2 * num_freqs)
        """
        v = 2 * torch.pi * x @ self.plane_wave_freqs
        # (n, num_freqs)
        return torch.cat([v.sin(), v.cos()], dim=-1)

    def non_equivariant_prediction(self, x: Tensor, element_indices: Tensor, wyckoff_indices: Tensor):
        """
        Args:
            x: torch.float
                shape (n_atoms, 3). Fractional coordinates.

        Returns:
            shape (n_atoms, 3) non-equivariant vectors
        """
        return self.mlp(self.plane_wave_fourier_features(x))

    def forward(self):
        raise NotImplementedError

    def predict_equivariant_vectors(
        self,
        frac_coords: Tensor,
        element_indices: Tensor,
        wyckoff_indices: Tensor,
        space_group_indices: Tensor,
        n_atoms_per_xtal: Tensor,
    ):
        """
        Symmetrization as mean(A^-1 @ f(Ax + t))

        Args:
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

        Returns:

        """
        device = frac_coords.device
        frac_coords = frac_coords % 1.0
        element_indices = torch.zeros_like(wyckoff_indices)  # todo

        (
            A_inv_ops,                      # (n_general_wyckoff_ops, 3, 3)
            t_ops,                          # (n_general_wyckoff_ops, 1, 3)
            inverse_indices,                # (n_general_wyckoff_ops,)
            map_conventional_to_asu_atom,   # (n_general_wyckoff_ops,). Ascending.
            conventional_wyckoff_indices,   # (n_unique_conventional_atoms,)
            conventional_element_indices,   # (n_unique_conventional_atoms,)
            frac_coords_of_conv_atoms,      # (n_unique_conventional_atoms, 3)
            _,
        ) = get_space_group_ops_and_conventional_atoms(
            frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
        )

        # Scatter mean
        src = torch.bmm(
            self.non_equivariant_prediction(
                frac_coords_of_conv_atoms,
                conventional_element_indices,
                conventional_wyckoff_indices
            )[inverse_indices].view(-1, 1, 3),
            A_inv_ops,
        ).squeeze(dim=1)
        # (n_general_wyckoff_ops, 3)
        if self.training and self.subsample_group_operations:
            # -- Uniformly sub-sample 1 group operation per atom
            # todo: sub-sample k group elements per atom

            # Identify first occurrence of each unique value
            first_unique_mask = torch.cat(
                [
                    torch.tensor([True], device=device),
                    map_conventional_to_asu_atom[1:] != map_conventional_to_asu_atom[:-1]
                ]
            )  # (n_general_wyckoff_ops,)

            # Shuffle within each atom orbit to get a random group op per atom
            randperm = torch.randperm(n=map_conventional_to_asu_atom.shape[0], device=device)
            # (n_general_wyckoff_ops,)
            selected_indices = torch.argmax(
                randperm *
                (map_conventional_to_asu_atom == map_conventional_to_asu_atom.unsqueeze(1)),
                dim=1
            )[first_unique_mask]
            # (n_asu_atoms,)

            # Select 1 group op per atom
            src = src[selected_indices]
            map_conventional_to_asu_atom = map_conventional_to_asu_atom[selected_indices]

        vector_field = scatter(
            src=src,
            index=map_conventional_to_asu_atom,
            dim=0,
            dim_size=frac_coords.shape[0],
            reduce='mean',
        )  # (n_asu_atoms, 3)
        return vector_field

@torch.no_grad()
def test_equivariance(
    model,
    frac_coords,
    element_indices,
    wyckoff_indices,
    space_group_indices,
    n_atoms_per_xtal,
):
    """Equivariance means f(gx)=gf(x)"""
    device = frac_coords.device
    fx = model.predict_equivariant_vectors(
        frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
    )
    batch_atom_offsets = torch.cat(
        [
            torch.cumsum(n_atoms_per_xtal, dim=0) - n_atoms_per_xtal,
            n_atoms_per_xtal.sum().long().view(1)
        ], dim=0
    )
    for i in range(len(space_group_indices)):
        space_group_number = 1 + int(space_group_indices[i])
        pyxtal_space_group = PyxtalGroup(space_group_number, style="pyxtal")
        general_wyckoff_position = pyxtal_space_group.Wyckoff_positions[0]
        for symmop in general_wyckoff_position.ops:
            rotation = torch.tensor(
                symmop.rotation_matrix, dtype=torch.float, device=device
            ).transpose(0, 1)
            # (3, 3)
            translation = torch.tensor(
                symmop.translation_vector, dtype=torch.float, device=device
            )[None, ...]
            # (1, 3)

            fgx = model.predict_equivariant_vectors(
                frac_coords @ rotation + translation,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
            )[batch_atom_offsets[i]:batch_atom_offsets[i+1]]
            gfx = fx[batch_atom_offsets[i]:batch_atom_offsets[i+1]] @ rotation
            assert torch.all(torch.isclose(fgx, gfx, atol=1e-6))


@torch.no_grad()
def test_equivariance_in_every_wyckoff(model):
    import cloudpickle
    from utils.crystal_utils import uniformly_sample_point_in_asu_wyckoff_site
    from utils.io_utils import SHAPE_DECOMP_DICT_PATH
    from utils.wyckoff_shape_decomp_dict_builder import build_wyckoff_shape_decomposition_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_wyckoff_shape_decomposition_dict()
    with open(SHAPE_DECOMP_DICT_PATH.as_posix(), "rb") as f:
        asu_quadrature_dict = cloudpickle.load(f)

    asu_wyckoff_dict: Dict = global_vars.asu_wyckoff_dict
    for sg_num in tqdm(range(1, 231)):
        space_group_dict = asu_wyckoff_dict[str(sg_num)]
        wyckoff_letters = space_group_dict["ordered_wyckoff_letters"]

        # Uniformly sample point in asu wyckoff site
        (
            rand_x, wyckoff_shape_indices
        ) = uniformly_sample_point_in_asu_wyckoff_site(
            space_group_numbers=[str(sg_num)] * len(wyckoff_letters),
            wyckoff_letters=wyckoff_letters,
            dictionary_of_wyckoffs_in_asu=asu_wyckoff_dict,
            dictionary_of_wyckoff_shape_decompositions=asu_quadrature_dict,
            device=device,
            n_samples_per_wyckoff=1,
            return_sampled_wyckoff_shape_indices=True,
        )
        # (n_wyckoffs, n_samples_per_wyckoff, 3)
        # (n_wyckoffs, n_samples_per_wyckoff)
        frac_coords = rand_x.view(-1, 3)
        # (n_wyckoffs, 3)

        # Predict flows
        fx = model.predict_equivariant_vectors(
            frac_coords,
            None,
            wyckoff_indices=torch.tensor(list(range(len(wyckoff_letters))), device=device),
            space_group_indices=torch.tensor([sg_num - 1], device=device),
            n_atoms_per_xtal=torch.tensor([len(wyckoff_letters)], device=device),
        )  # (n_wyckoffs, 3)

        pyxtal_space_group = PyxtalGroup(sg_num, style="pyxtal")
        general_wyckoff_position = pyxtal_space_group.Wyckoff_positions[0]
        for symmop in general_wyckoff_position.ops:
            rotation = torch.tensor(
                symmop.rotation_matrix, dtype=torch.float, device=device
            )
            # (3, 3)
            translation = torch.tensor(
                symmop.translation_vector, dtype=torch.float, device=device
            )[None, ...]
            # (1, 3)

            fgx = model.predict_equivariant_vectors(
                frac_coords @ rotation.T + translation,
                None,
                wyckoff_indices=torch.tensor(list(range(len(wyckoff_letters))), device=device),
                space_group_indices=torch.tensor([sg_num - 1], device=device),
                n_atoms_per_xtal=torch.tensor([len(wyckoff_letters)], device=device),
            )
            gfx = fx @ rotation.T
            assert torch.all(torch.isclose(fgx, gfx, atol=1e-6))


@torch.no_grad()
def test_equivariant_vectors_live_in_wyckoff_subspaces():
    """
    Sample x in every shape of every Wyckoff. Predict flows. Assert that
    (flow == flow @ projection).
    """
    import cloudpickle
    from utils.crystal_utils import uniformly_sample_point_in_asu_wyckoff_site
    from utils.wyckoff_shape_decomp_dict_builder import build_wyckoff_shape_decomposition_dict, PROJECT_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quadrature_rule_order = 10
    build_wyckoff_shape_decomposition_dict(quadrature_rule_order)
    quadrature_path = PROJECT_DIR + "/media/asu_quadrature_" + str(quadrature_rule_order) + ".pkl"
    with open(quadrature_path, "rb") as f:
        asu_quadrature_dict = cloudpickle.load(f)

    model = SpaceGroupEquivariantVectorField()
    asu_wyckoff_dict: Dict = global_vars.asu_wyckoff_dict
    for sg_num in tqdm(range(1, 231)):
        space_group_dict = asu_wyckoff_dict[str(sg_num)]
        wyckoff_letters = space_group_dict["ordered_wyckoff_letters"]

        # Uniformly sample point in asu wyckoff site
        (
            rand_x, wyckoff_shape_indices
        ) = uniformly_sample_point_in_asu_wyckoff_site(
            space_group_numbers=[str(sg_num)] * len(wyckoff_letters),
            wyckoff_letters=wyckoff_letters,
            dictionary_of_wyckoffs_in_asu=asu_wyckoff_dict,
            dictionary_of_wyckoff_shape_decompositions=asu_quadrature_dict,
            device=device,
            n_samples_per_wyckoff=1,
            return_sampled_wyckoff_shape_indices=True,
        )
        # (n_wyckoffs, n_samples_per_wyckoff, 3)
        # (n_wyckoffs, n_samples_per_wyckoff)
        rand_x = rand_x.view(-1, 3)
        # (n_wyckoffs, 3)

        # Predict flows
        flows = model.predict_equivariant_vectors(
            rand_x,
            None,
            wyckoff_indices=torch.tensor(list(range(len(wyckoff_letters))), device=device),
            space_group_indices=torch.tensor([sg_num - 1], device=device),
            n_atoms_per_xtal=torch.tensor([len(wyckoff_letters)], device=device),
        )  # (n_wyckoffs, 3)

        # Get Wyckoff shape projection matrices
        wyckoff_indices = list(range(len(wyckoff_letters)))
        space_group_indices_per_asu_atom = torch.tensor([sg_num - 1] * len(wyckoff_letters), device=device)
        projection_matrices = global_vars.noise_projection_matrices.to(device)[
            space_group_indices_per_asu_atom,
            wyckoff_indices,
            wyckoff_shape_indices.view(-1),
        ]
        # (n_wyckoffs, 3, 3)
        projected_flows = torch.bmm(flows[:, None, :], projection_matrices).view(-1, 3)
        # (n_wyckoffs, 3)
        assert torch.all(torch.isclose(flows, projected_flows)), f"{torch.abs(flows - projected_flows)}"


def get_training_crystals(n=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = AsymmetricUnitDataset(split="train", name="mp_20")
    xtals_to_overfit_on = []
    frac_coords = []
    element_indices = []
    wyckoff_indices = []
    space_group_indices = []
    n_atoms_per_xtal = []
    wyckoff_shape_indices = []
    for i_xtal in dataset.data:
        xtal = i_xtal.to_ASUCrystal()
        wyckoff_dimensionalities = global_vars.wyckoff_dimension_tensor.to(device)[
            (xtal.space_group_number - 1).view(1).expand(xtal.num_atoms),
            xtal.wyckoff_indices
        ]

        atom_is_zero_dim = (wyckoff_dimensionalities == 0)
        atom_is_one_dim = (wyckoff_dimensionalities == 1)
        atom_is_two_dim = (wyckoff_dimensionalities == 2)
        atom_is_three_dim = (wyckoff_dimensionalities == 3)
        if (
            torch.any(atom_is_zero_dim) and
            torch.any(atom_is_one_dim) and
            torch.any(atom_is_two_dim) and
            torch.any(atom_is_three_dim)
        ):
            print(xtal)
            xtals_to_overfit_on.append(xtal)
            frac_coords.append(xtal.conventional_frac_coords)
            element_indices.append(xtal.element_indices)
            wyckoff_indices.append(xtal.wyckoff_indices)
            space_group_indices.append(xtal.space_group_number - 1)
            n_atoms_per_xtal.append(xtal.num_atoms)
            wyckoff_shape_indices.append(xtal.wyckoff_shape_indices)
            if len(xtals_to_overfit_on) >= n:
                break

    frac_coords = torch.cat(frac_coords, dim=0)
    element_indices = torch.cat(element_indices, dim=0)
    wyckoff_indices = torch.cat(wyckoff_indices, dim=0)
    space_group_indices = torch.tensor(space_group_indices, dtype=torch.long)
    n_atoms_per_xtal = torch.tensor(n_atoms_per_xtal, dtype=torch.long)
    wyckoff_shape_indices = torch.cat(wyckoff_shape_indices, dim=0)
    return (
        frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        wyckoff_shape_indices,
    )


def main():
    (
        frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        wyckoff_shape_indices,
    ) = get_training_crystals(n=2)

    model = SpaceGroupEquivariantVectorField()
    vectors = model.predict_equivariant_vectors(
        frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
    )

    test_equivariance(
        model,
        frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
    )
    test_equivariance_in_every_wyckoff(model)
    test_equivariant_vectors_live_in_wyckoff_subspaces()


if __name__ == "__main__":
    main()
