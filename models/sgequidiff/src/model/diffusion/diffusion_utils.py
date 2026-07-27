import torch
import torch.nn as nn
import torch_scatter
import numpy as np
from scipy.spatial import ConvexHull

from aliases import *
import global_vars
from custom_dataset import AsymmetricUnitDataset
from constants import MAX_WYCKOFF_SITES, OFFSET_LIST
from utils.crystal_utils import vmappable_inside
from utils.data_utils import (
    batched_convert_asu_frac_coords_to_primitive_cartesian_coords,
    lattice_params_to_matrix_torch,
)


"""
Args:
    x: shape (n_points, n_lattice_translations, 3)
    hull_equations: shape (n_points, n_bounds, 4)
Returns:
    shape (n_points, n_lattice_translations) boolean Tensor
"""
is_inside_linear_wyckoff_subspace: Callable = torch.vmap(
    torch.vmap(
        vmappable_inside, in_dims=0, out_dims=0  # vmap n_atoms
    ),
    in_dims=(1, None), out_dims=1  # vmap lattice translations
)


def get_infinite_wyckoff_shape_hull_equations(
    device: Union[str, torch.device] = "cpu"
) -> Tuple[Tensor, Tensor]:
    """
    Get padded tensors of inequalities for inside/outside tests on the INFINITE
    1/2D Wyckoff shapes as well as each 0D Wyckoff point in the ASU.

    Returns:
        padded_hull_equations: torch.float
            shape (n_space_groups, max_wyckoffs, max_shapes, max_shape_bounds, 4)
            where the last dimension is used as in utils.crystal_utils.inside()
        mask_padded_hull_equations: torch.bool
            shape (n_space_groups, max_wyckoffs, max_shapes) tensor which is
            True to indicate a shape exists at that index.
    """
    asu_wyckoff_dict: Dict = global_vars.asu_wyckoff_dict
    max_num_shape_bounds = max(
        2,  # 1D
        5,  # 2D
        global_vars.max_simplicial_hull_facets,  # 3D
    )

    # Initialize hull equations with padding that makes the inside test
    # always evaluate to True
    padded_hull_equations = -1.0 * nn.functional.one_hot(
        torch.tensor(3, device=device), num_classes=4
    )[None, None, None, None, :].repeat(
        230,
        MAX_WYCKOFF_SITES,
        global_vars.max_shapes_per_wyckoff,
        max_num_shape_bounds,
        1,
    )
    # (n_space_groups, n_wyckoffs, max_shapes, n_linear_shape_bounds, 4)

    # Mask is True to indicate a shape exists at that index
    mask_padded_hull_equations = torch.zeros(
        230, MAX_WYCKOFF_SITES, global_vars.max_shapes_per_wyckoff,
        dtype=torch.bool, device=device
    )
    # (n_space_groups, n_wyckoffs, max_shapes)

    for space_group_number in range(1, 231):
        space_group_dict = asu_wyckoff_dict[str(space_group_number)]
        for wyckoff_index, wyckoff_letter in enumerate(
            space_group_dict["ordered_wyckoff_letters"]
        ):
            wyckoff_dict = space_group_dict[wyckoff_letter]
            wyckoff_dim: int = wyckoff_dict["dim"]
            if wyckoff_dim == 0:
                shape_index = 0
                mask_padded_hull_equations[
                    space_group_number - 1, wyckoff_index, shape_index
                ] = True

                vertex: Tensor = wyckoff_dict["vertices"].astype("float64").reshape(-1)
                # (3,)

                # Put a small bounding box around the Wyckoff point
                epsilon = 1e-4
                hull_equations = np.zeros((6, 4))
                hull_equations[:3, :3] = np.eye(3)
                hull_equations[3:, :3] = -np.eye(3)
                hull_equations[:3, -1] = -(vertex + epsilon)
                hull_equations[3:, -1] = -(-vertex + epsilon)
                num_bounds: int = hull_equations.shape[0]

                padded_hull_equations[
                    space_group_number - 1,
                    wyckoff_index,
                    shape_index,
                    :num_bounds,
                    :
                ] = torch.tensor(
                    hull_equations, dtype=torch.float, device=device
                )
            elif wyckoff_dim == 1:
                for shape_index, line_segment in enumerate(
                    wyckoff_dict["vertices"].astype("float64")
                ):
                    # Put a rectangular prism with height/width of epsilon
                    # around the Wyckoff line. It is infinite along the line.
                    mask_padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                    ] = True

                    line_direction = line_segment[1] - line_segment[0]
                    line_direction /= np.linalg.norm(line_direction)
                    # (3,)

                    # Get a normal vector to the line segment
                    normal_direction1 = np.cross(
                        line_direction, np.random.rand(3)
                    )[None, :]
                    normal_direction1 /= np.linalg.norm(normal_direction1)
                    # (1, 3)

                    # Get a second normal
                    normal_direction2 = np.cross(
                        line_direction, normal_direction1
                    )
                    normal_direction2 /= np.linalg.norm(normal_direction2)
                    # (1, 3)

                    # Assert mutually orthonormal
                    assert (
                        np.abs(np.sum(normal_direction1 * line_direction)) < 1e-10 and
                        np.abs(np.sum(normal_direction2 * line_direction)) < 1e-10 and
                        np.abs(np.sum(normal_direction1 * normal_direction2)) < 1e-10
                    )

                    # -----
                    # Get hull equations for the rectangular prism
                    line_direction = line_direction.reshape(3, 1)
                    projection_matrix = line_direction @ line_direction.T
                    # (3, 3)

                    hull_equations = np.zeros((4, 4))
                    i_minus_p = np.eye(3) - projection_matrix
                    # (3, 3)
                    hull_equations[:, :3] = np.concatenate(
                        [
                            normal_direction1 @ i_minus_p,  # (1, 3)
                            -normal_direction1 @ i_minus_p,
                            normal_direction2 @ i_minus_p,
                            -normal_direction2 @ i_minus_p,
                        ], axis=0
                    )
                    hull_equations[:, -1] = -1 * np.concatenate(
                        [
                            1e-4 + normal_direction1 @ i_minus_p @ line_segment[0].reshape(3, 1),
                            1e-4 - normal_direction1 @ i_minus_p @ line_segment[0].reshape(3, 1),
                            1e-4 + normal_direction2 @ i_minus_p @ line_segment[0].reshape(3, 1),
                            1e-4 - normal_direction2 @ i_minus_p @ line_segment[0].reshape(3, 1),
                        ]
                    ).reshape(4)
                    num_facets: int = hull_equations.shape[0]
                    assert num_facets <= padded_hull_equations.shape[3]
                    # ---

                    padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                        :num_facets,
                        :
                    ] = torch.tensor(
                        hull_equations, dtype=torch.float, device=device
                    )
            elif wyckoff_dim == 2:
                for shape_index, polygon_vertices in enumerate(
                    wyckoff_dict["vertices"]
                ):
                    # Put a rectangular prism with height of epsilon around the
                    # Wyckoff plane. It is infinite in the plane.
                    mask_padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                    ] = True

                    polygon_vertices = polygon_vertices.astype("float64")

                    # Get plane normal
                    a = polygon_vertices[0]
                    b = polygon_vertices[1]
                    c = polygon_vertices[2]
                    ab = b - a
                    ac = c - a
                    polygon_plane_normal = np.cross(ab, ac)
                    polygon_plane_normal /= np.linalg.norm(polygon_plane_normal)
                    # (3,)

                    # ---------
                    polygon_plane_normal = polygon_plane_normal.reshape(3, 1)
                    projection_matrix = global_vars._project_onto_2d_subspace(
                        torch.tensor(polygon_vertices)
                    ).numpy().T
                    # (3, 3)
                    i_minus_p = np.eye(3) - projection_matrix

                    hull_equations = np.zeros((2, 4))
                    hull_equations[:, :3] = np.concatenate(
                        [
                            polygon_plane_normal.T @ i_minus_p,
                            -polygon_plane_normal.T @ i_minus_p,
                        ], axis=0
                    )
                    hull_equations[:, -1] = -1.0 * np.concatenate(
                        [
                            1e-4 + polygon_plane_normal.T @ i_minus_p @ polygon_vertices[0].reshape(3, 1),
                            1e-4 - polygon_plane_normal.T @ i_minus_p @ polygon_vertices[0].reshape(3, 1),
                        ], axis=0
                    ).reshape(2)
                    # ---------

                    num_facets: int = hull_equations.shape[0]
                    assert num_facets <= padded_hull_equations.shape[3]
                    padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                        :num_facets,
                        :
                    ] = torch.tensor(
                        hull_equations, dtype=torch.float, device=device
                    )

    return padded_hull_equations, mask_padded_hull_equations


def get_wyckoff_shape_hull_equations(device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
    """
    Get padded tensors of inequalities for inside/outside tests on the
    1/2/3D Wyckoff shapes.

    Returns:
        padded_hull_equations: torch.float
            shape (n_space_groups, max_wyckoffs, max_shapes, max_shape_bounds, 4)
            where the last dimension is used as in utils.crystal_utils.inside()
        mask_padded_hull_equations: torch.bool
            shape (n_space_groups, max_wyckoffs, max_shapes) tensor which is
            True to indicatea shape exists at that index.
    """
    asu_wyckoff_dict: Dict = global_vars.asu_wyckoff_dict
    max_num_shape_bounds = max(
        2,  # 1D
        5,  # 2D
        global_vars.max_simplicial_hull_facets,  # 3D
    )

    # Initialize hull equations with padding that makes the inside test
    # always evaluate to True
    padded_hull_equations = -1.0 * nn.functional.one_hot(
        torch.tensor(3, device=device), num_classes=4
    )[None, None, None, None, :].repeat(
        230,
        MAX_WYCKOFF_SITES,
        global_vars.max_shapes_per_wyckoff,
        max_num_shape_bounds,
        1,
    )
    # (n_space_groups, n_wyckoffs, max_shapes, n_linear_shape_bounds, 4)

    # Mask is True to indicate a shape exists at that index
    mask_padded_hull_equations = torch.zeros(
        230, MAX_WYCKOFF_SITES, global_vars.max_shapes_per_wyckoff,
        dtype=torch.bool, device=device
    )
    # (n_space_groups, n_wyckoffs, max_shapes)

    for space_group_number in range(1, 231):
        space_group_dict = asu_wyckoff_dict[str(space_group_number)]
        for wyckoff_index, wyckoff_letter in enumerate(
            space_group_dict["ordered_wyckoff_letters"]
        ):
            wyckoff_dict = space_group_dict[wyckoff_letter]
            wyckoff_dim: int = wyckoff_dict["dim"]
            if wyckoff_dim == 0:
                shape_index = 0
                mask_padded_hull_equations[
                    space_group_number - 1, wyckoff_index, shape_index
                ] = True

                vertex: Tensor = wyckoff_dict["vertices"].astype("float64").reshape(-1)
                # (3,)

                # Put a small bounding box around the Wyckoff point
                epsilon = 1e-4
                hull_equations = np.zeros((6, 4))
                hull_equations[:3, :3] = np.eye(3)
                hull_equations[3:, :3] = -np.eye(3)
                hull_equations[:3, -1] = -(vertex + epsilon)
                hull_equations[3:, -1] = -(-vertex + epsilon)
                num_bounds: int = hull_equations.shape[0]

                padded_hull_equations[
                    space_group_number - 1,
                    wyckoff_index,
                    shape_index,
                    :num_bounds,
                    :
                ] = torch.tensor(
                    hull_equations, dtype=torch.float, device=device
                )
            elif wyckoff_dim == 1:
                for shape_index, line_segment in enumerate(
                    wyckoff_dict["vertices"].astype("float64")
                ):
                    mask_padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                    ] = True

                    line_direction = line_segment[1] - line_segment[0]
                    line_direction /= np.linalg.norm(line_direction)
                    # (3,)

                    # Get a normal vector to the line segment
                    normal_direction1 = np.cross(
                        line_direction, np.random.rand(3)
                    )[None, :]
                    normal_direction1 /= np.linalg.norm(normal_direction1)
                    # (1, 3)

                    # Get a second normal
                    normal_direction2 = np.cross(
                        line_direction, normal_direction1
                    )
                    normal_direction2 /= np.linalg.norm(normal_direction2)
                    # (1, 3)

                    # Assert mutually orthonormal
                    assert (
                        np.abs(np.sum(normal_direction1 * line_direction)) < 1e-10 and
                        np.abs(np.sum(normal_direction2 * line_direction)) < 1e-10 and
                        np.abs(np.sum(normal_direction1 * normal_direction2)) < 1e-10
                    )

                    # Construct bounding rectangular prism around the line
                    # segment with length equal to the line segment length
                    perturbing_direction1 = 1e-4 * normal_direction1
                    perturbing_direction2 = 1e-4 * normal_direction2
                    bounding_polytope = np.concatenate([
                        line_segment + (perturbing_direction1 + perturbing_direction2),
                        line_segment + (perturbing_direction1 - perturbing_direction2),
                        line_segment + (-perturbing_direction1 + perturbing_direction2),
                        line_segment + (-perturbing_direction1 - perturbing_direction2)
                    ], axis=0)
                    # (8, 3)
                    hull = ConvexHull(bounding_polytope)
                    num_facets: int = hull.equations.shape[0]

                    padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                        :num_facets,
                        :
                    ] = torch.tensor(
                        hull.equations, dtype=torch.float, device=device
                    )
            elif wyckoff_dim == 2:
                for shape_index, polygon_vertices in enumerate(
                    wyckoff_dict["vertices"]
                ):
                    mask_padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                    ] = True

                    polygon_vertices = polygon_vertices.astype("float64")
                    # --
                    # To get bounding planes of the 2D polygon in 3D space,
                    # construct a 3D polytope bounding the Wyckoff polygon using
                    # the set of 3D vertices,
                    #   (
                    #       Wyckoff polygon vertices in 3D
                    #       +/- epsilon * polygon_plane_normal
                    #   ).
                    # --
                    # Get plane normal
                    a = polygon_vertices[0]
                    b = polygon_vertices[1]
                    c = polygon_vertices[2]
                    ab = b - a
                    ac = c - a
                    polygon_plane_normal = np.cross(ab, ac)
                    polygon_plane_normal /= np.linalg.norm(polygon_plane_normal)
                    # (3,)

                    # Get bounding polytope and its bounding plane equations
                    bounding_polytope = np.concatenate(
                        [
                            polygon_vertices + 1e-4 * polygon_plane_normal,
                            polygon_vertices - 1e-4 * polygon_plane_normal
                        ], axis=0
                    )  # (2 * n_polygon_vertices, 3)
                    hull = ConvexHull(bounding_polytope)
                    hull_equations = hull.equations

                    num_facets: int = hull_equations.shape[0]
                    padded_hull_equations[
                        space_group_number - 1,
                        wyckoff_index,
                        shape_index,
                        :num_facets,
                        :
                    ] = torch.tensor(
                        hull_equations, dtype=torch.float, device=device
                    )
            elif wyckoff_dim == 3:
                shape_index = 0  # only one 3D shape
                mask_padded_hull_equations[
                    space_group_number - 1,
                    wyckoff_index,
                    shape_index,
                ] = True

                hull = ConvexHull(wyckoff_dict["vertices"].astype("float64"))
                num_facets: int = hull.equations.shape[0]
                padded_hull_equations[
                    space_group_number - 1,
                    wyckoff_index,
                    shape_index,
                    :num_facets,
                    :
                ] = torch.tensor(
                    hull.equations, dtype=torch.float, device=device
                )

    return padded_hull_equations, mask_padded_hull_equations


@torch.no_grad()
def p_asu_wrapped_normal(
    noisy_frac_coord: Tensor,           # (n_asu_atoms, 3)
    conventional_frac_coords: Tensor,   # (n_conventional_atoms, 3)
    map_conventional_to_asu_frac_coords: Tensor,    # (n_conventional_atoms,)
    n_lattice_translations: int = 5,
    sigma: Union[float, Tensor] = 1.0,  # (n_conventional_atoms, 1)
):
    """
    Put isotropic Gaussians centered at each equivalent point in
    conventional_frac_coords and sum. Excludes pre-factor 1/2pi*sigma.
    """
    device = conventional_frac_coords.device

    translations_1d = torch.arange(-n_lattice_translations, n_lattice_translations+1, device=device)
    # (n_lattice_translations_per_dim,)
    translations = torch.cartesian_prod(translations_1d, translations_1d, translations_1d)
    # (n_lattice_translations, 3)

    # - For k lattice translations per dim, sum over k^3 grid
    noisy_x_minus_gt_xs = (
        noisy_frac_coord[map_conventional_to_asu_frac_coords][:, None, :]
        - (conventional_frac_coords[:, None, :] + translations[None, :, :])
    )
    # (n_conventional_atoms, n_lattice_translations, 3)

    """
    The next code block (1) splits the below operation into a few lines so 
    pytorch can potentially release intermediate values sooner, and (2) uses 
    in-place operations which is allowed since this function is wrapped with a
    torch.no_grad().
    
    p = torch.sum(
        torch.exp(-(noisy_x_minus_gt_xs ** 2).sum(dim=-1) / 2 / sigma ** 2),
        # (n_conventional_atoms, n_lattice_translations)
        dim=1
    )  # (n_conventional_atoms,)
    """
    diff_sq = (noisy_x_minus_gt_xs.pow_(2)).sum(dim=-1)
    # (n_conventional_atoms, n_lattice_translations)
    gaussian = diff_sq.neg_().div_(2 * sigma ** 2).exp_()
    # (n_conventional_atoms, n_lattice_translations)
    p = torch.sum(gaussian, dim=1)
    # (n_conventional_atoms,)

    # Sum over space group ops
    p = torch.scatter_add(
        input=torch.zeros(noisy_frac_coord.shape[0], device=device),
        dim=0,
        index=map_conventional_to_asu_frac_coords,
        src=p,
    )
    # (n_asu_atoms,)
    return p


@torch.no_grad()
def d_log_p_asu_wrapped_normal(
    noisy_frac_coord: Tensor,               # (n_asu_atoms, 3)
    conventional_frac_coords: Tensor,       # (n_conventional_atoms, 3)
    map_conventional_to_asu_frac_coords: Tensor,    # (n_conventional_atoms,)
    n_lattice_translations: int = 5,
    sigma: Union[float, Tensor] = 1.0,  # (n_asu_atoms,)
):
    if isinstance(sigma, float):
        sigma_conventional = torch.tensor([[sigma]], device=noisy_frac_coord.device)
    elif isinstance(sigma, Tensor):
        sigma_conventional = sigma[map_conventional_to_asu_frac_coords][:, None]
        # (n_conventional_atoms, 1)

    device = conventional_frac_coords.device

    translations_1d = torch.arange(-n_lattice_translations, n_lattice_translations+1, device=device)
    # (n_lattice_translations_per_dim,)
    translations = torch.cartesian_prod(translations_1d, translations_1d, translations_1d)
    # (n_lattice_translations, 3)

    # -- For k lattice translations per dim, sum over k^3 grid
    noisy_x_minus_gt_xs = (
        noisy_frac_coord[map_conventional_to_asu_frac_coords][:, None, :]
        - (conventional_frac_coords[:, None, :] + translations[None, :, :])
    )
    # (n_conventional_atoms, n_lattice_translations, 3)

    """
    The next code block (1) splits the below operation into a few lines so 
    pytorch can potentially release intermediate values sooner, and (2) uses 
    in-place operations which is allowed since this function is wrapped with a
    torch.no_grad().
    
    numerator = -1.0 * torch.sum(
        torch.exp(
            -(noisy_x_minus_gt_xs ** 2).sum(dim=-1, keepdims=True) / 2 / sigma_conventional[..., None] ** 2
        ) * noisy_x_minus_gt_xs,
        # (n_conventional_atoms, n_lattice_translations, 3)
        dim=1
    )  # (n_conventional_atoms, 3)
    """
    diff_sq = (noisy_x_minus_gt_xs ** 2).sum(dim=-1, keepdims=True)
    # (n_conventional_atoms, n_lattice_translations, 1)
    gaussian = diff_sq.neg_().div_(2 * sigma_conventional[..., None] ** 2).exp_()
    # (n_conventional_atoms, n_lattice_translations, 1)
    numerator = torch.sum(gaussian * noisy_x_minus_gt_xs, dim=1).neg_()

    # Sum over space group ops in the unit cell
    numerator = torch.scatter_add(
        input=torch.zeros(noisy_frac_coord.shape[0], 3, device=device),
        dim=0,
        index=map_conventional_to_asu_frac_coords[:, None].expand(-1, 3),
        src=numerator,
    )
    # (n_asu_atoms, 3)

    denominator = p_asu_wrapped_normal(
        noisy_frac_coord,
        conventional_frac_coords,
        map_conventional_to_asu_frac_coords,
        n_lattice_translations,
        sigma_conventional,
    ) * (sigma ** 2)
    # (n_asu_atoms,)
    return numerator / denominator[:, None]  # (n_asu_atoms, 3)


"""
Args:
    x: shape (n_points, num_cell_images, 3)
    hull_equations: shape (n_points, n_shapes, n_linear_shape_bounds, 4)
Returns:
    shape (n_points, num_cell_images, n_shapes) boolean Tensor
"""
vmap_atoms_are_in_hull: Callable = torch.vmap(
    torch.vmap(
        torch.vmap(
            vmappable_inside, in_dims=0, out_dims=0  # vmap n_points
        ),
        in_dims=(None, 1), out_dims=1  # vmap n_shapes
    ),
    in_dims=(1, None), out_dims=1  # vmap supercell images
)


def wrap_frac_coords_into_asu(
    frac_coords: Tensor,
    wyckoff_indices: Tensor,
    space_group_indices: Tensor,
    num_atoms_per_asu: Tensor,
    hull_equations: Tensor,
    hull_equations_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Modulo group operations that take a point out of the canonical ASU.

    Args:
        space_group_indices: torch.long
            (n_crystals,)
        frac_coords: torch.float
            (n_asu_atoms, 3)
        wyckoff_indices: torch.long
            (n_asu_atoms,)
        num_atoms_per_asu: torch.long
            (n_crystals,)
        hull_equations: torch.float
            (n_asu_atoms, max_shapes, max_linear_shape_bounds, 4)
        hull_equations_mask: torch.bool
            (n_asu_atoms, max_shapes)

    Returns:
        wrapped_asu_frac_coords: torch.float
            shape (n_asu_atoms, 3)
        wrapped_asu_wyckoff_shape_indices: torch.long
            shape (n_asu_atoms,)
    """
    device = frac_coords.device
    n_asu_atoms_in_batch: int = frac_coords.shape[0]
    frac_coords = frac_coords % 1.0

    # - 1. Orbit coords to conventional cell
    (
        conventional_cell_frac_coords,    # (n_conventional_atoms, 3)
        _,
        conventional_wyckoff_indices,     # (n_conventional_atoms,)
        num_conventional_atoms_per_xtal,  # (batch_size,)
        _,
        map_conventional_to_asu_coord,     # (n_conventional_atoms,)
        _,
    ) = batched_convert_asu_frac_coords_to_primitive_cartesian_coords(
        asu_frac_coords=frac_coords,
        asu_element_indices=torch.zeros_like(wyckoff_indices),
        asu_wyckoff_indices=wyckoff_indices,
        n_coords_per_asu=num_atoms_per_asu,
        conventional_lattice_matrix=torch.zeros(
            space_group_indices.shape[0], 3, 3, dtype=torch.float, device=device
        ),
        space_group_indices=space_group_indices,
        device=device,
        return_cartesian_coords=False,  # get fractional
        get_primitive_cell=False,  # get conventional
    )

    # - 2. Get 3x3x3 supercell
    supercell_frac_translations = torch.tensor(
        OFFSET_LIST, dtype=torch.float, device=device
    )  # (27 supercell images, 3)
    supercell_frac_coords = (
        conventional_cell_frac_coords[:, None, :]
        + supercell_frac_translations[None, ...]
    )  # (n_conventional_atoms, n_lattice_translations, 3)

    # -
    # 3. Use inside/outside tests to determine which Wyckoff shapes are
    # occupied, and apply periodic boundary conditions to the noisy ASU
    # fractional coordinates
    # -
    hull_equations = hull_equations[map_conventional_to_asu_coord]
    # (n_conventional_atoms, max_shapes, max_linear_shape_bounds, 4)
    wyckoff_shape_exists: Tensor = hull_equations_mask[map_conventional_to_asu_coord]
    # (n_conventional_atoms, max_shapes)

    supercell_frac_coord_is_in_wyckoff_shape: Tensor = (
        vmap_atoms_are_in_hull(
            supercell_frac_coords, hull_equations, epsilon=-1e-5
        )  # (n_conventional_atoms, n_lattice_translations, max_shapes)
        & wyckoff_shape_exists[:, None, :]
    )  # (n_conventional_atoms, n_lattice_translations, max_shapes)
    supercell_frac_coord_is_in_any_wyckoff_shape = torch.any(
        supercell_frac_coord_is_in_wyckoff_shape, dim=-1
    )  # (n_conventional_atoms, n_lattice_translations)
    conventional_coord_has_image_in_any_wyckoff_shape = torch.any(
        supercell_frac_coord_is_in_any_wyckoff_shape, dim=-1
    )  # (n_conventional_atoms,)

    # Break ties with scatter_max in case >1 conventional atoms originating
    # from 1 ASU atom are in Wyckoff shapes by numerical precision issues
    _, indices_of_conv_atoms_in_asu = torch_scatter.scatter_max(
        src=conventional_coord_has_image_in_any_wyckoff_shape.float(),
        index=map_conventional_to_asu_coord,
    )  # (n_asu_atoms,)
    assert indices_of_conv_atoms_in_asu.shape[0] == n_asu_atoms_in_batch

    # Reduce supercell of conventional atoms to supercell of ASU atoms
    supercell_frac_coords_in_asu = supercell_frac_coords[indices_of_conv_atoms_in_asu]
    # (n_asu_atoms, n_lattice_translations, 3)
    asu_atom_is_inside = supercell_frac_coord_is_in_wyckoff_shape[indices_of_conv_atoms_in_asu]
    # (n_asu_atoms, n_lattice_translations, max_shapes)
    lattice_translation_into_asu_idx = torch.argmax(
        supercell_frac_coord_is_in_any_wyckoff_shape[
            indices_of_conv_atoms_in_asu
        ].float(),  # (n_asu_atoms, n_lattice_translations)
        dim=-1
    )  # (n_asu_atoms,)

    # Get noisy atom coordinates and Wyckoff shape indices in ASU
    _atom_indices = torch.arange(n_asu_atoms_in_batch, device=device)
    wrapped_asu_frac_coords = supercell_frac_coords_in_asu[_atom_indices, lattice_translation_into_asu_idx]
    # (n_asu_atoms, 3)
    wrapped_asu_wyckoff_shape_indices = torch.argmax(
        asu_atom_is_inside[_atom_indices, lattice_translation_into_asu_idx].float(),
        dim=-1
    )  # (n_asu_atoms,)

    return wrapped_asu_frac_coords, wrapped_asu_wyckoff_shape_indices


def get_sg192_crystal():
    """
    Single-atom crystal in space group 192 with atom in Wyckoff position k. This
    Wyckoff position is 1D and has two disjoint line segments in the ASU,
    separated by a 0-dimensional Wyckoff position.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    space_group_dict: Dict = global_vars.asu_wyckoff_dict["192"]
    wyckoff_position_dict = space_group_dict["k"]

    # Place atom at midpoint of the first line segment.
    vertices = torch.tensor(
        wyckoff_position_dict["vertices"].astype("float32"), device=device
    )  # (num_line_segments, 2, 3)

    wyckoff_shape_indices = torch.tensor([0], dtype=torch.long, device=device)
    frac_coords = vertices[wyckoff_shape_indices].mean(dim=1)  # (1, 3)
    element_indices = torch.tensor([0], dtype=torch.long, device=device)
    wyckoff_indices = torch.tensor([10], dtype=torch.long, device=device)
    space_group_indices = torch.tensor([191], dtype=torch.long, device=device)
    n_atoms_per_xtal = torch.tensor([1], dtype=torch.long, device=device)
    return (
        frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        wyckoff_shape_indices,
    )


def get_training_crystals(n=1):
    """
    Args:
        n: number of crystals

    Returns:

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = AsymmetricUnitDataset(split="train", name="mp_20")
    xtals_to_overfit_on = []
    frac_coords = []
    element_indices = []
    wyckoff_indices = []
    space_group_indices = []
    n_atoms_per_xtal = []
    wyckoff_shape_indices = []
    lattice_lengths = []
    lattice_angles = []
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
            lattice_lengths.append(xtal.conventional_lattice_lengths)
            lattice_angles.append(xtal.conventional_lattice_angles)
            if len(xtals_to_overfit_on) >= n:
                break

    frac_coords = torch.cat(frac_coords, dim=0)
    element_indices = torch.cat(element_indices, dim=0)
    wyckoff_indices = torch.cat(wyckoff_indices, dim=0)
    space_group_indices = torch.tensor(space_group_indices, dtype=torch.long)
    n_atoms_per_xtal = torch.tensor(n_atoms_per_xtal, dtype=torch.long)
    wyckoff_shape_indices = torch.cat(wyckoff_shape_indices, dim=0)
    lattice_lengths = torch.stack(lattice_lengths)
    lattice_angles = torch.stack(lattice_angles)
    lattice_matrices = lattice_params_to_matrix_torch(
        lattice_lengths, lattice_angles
    )
    return (
        frac_coords,
        element_indices,
        wyckoff_indices,
        space_group_indices,
        n_atoms_per_xtal,
        wyckoff_shape_indices,
        lattice_matrices,
        lattice_lengths,
        lattice_angles,
    )


def get_space_group_ops_and_conventional_atoms(
    frac_coords: Tensor,
    element_indices: Tensor,
    wyckoff_indices: Tensor,
    space_group_indices: Tensor,
    n_atoms_per_xtal: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Get space group operations mod lattice translations and conventional
    unit cell atoms without duplicates.

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
        A_ops: torch.float
            (n_general_wyckoff_ops, 3, 3). O(3) component of space group
            elements.
        t_ops: torch.float
            (n_general_wyckoff_ops, 1, 3). Non-lattice translation component
            of space group elements.
        inverse_indices: torch.long
            (n_general_wyckoff_ops,). Useful because
                frac_coords_of_conv_atoms[inverse_indices]
            recovers the corresponding ordering of A_ops and t_ops.
        map_conventional_to_asu_atom: torch.long
            (n_general_wyckoff_ops,). Indexes which row of frac_coords each
            row of frac_coords_of_conv_atoms originated from.
        conventional_wyckoff_indices: torch.long
            (n_unique_conventional_atoms,).
        conventional_element_indices: torch.long
            (n_unique_conventional_atoms,).
        frac_coords_of_conv_atoms: torch.float
            (n_unique_conventional_atoms, 3).
    """
    batch_size: int = n_atoms_per_xtal.shape[0]
    device = frac_coords.device

    padded_general_wyckoff_matrices = global_vars.padded_general_wyckoff_matrices.to(device)[
        space_group_indices
    ]
    # (B, 192, 3, 3)
    padded_general_wyckoff_inv_matrices = global_vars.padded_inverse_general_wyckoff_matrices.to(device)[
        space_group_indices
    ]
    # (B, 192, 3, 3)
    padded_general_wyckoff_translations = global_vars.padded_general_wyckoff_translations.to(
        device
    )[space_group_indices]
    # (B, 192, 1, 3)
    padded_general_wyckoff_ops_mask = global_vars.padded_general_wyckoff_ops_mask.to(device)[
        space_group_indices
    ]
    # (B, 192)
    general_wyckoff_multiplicity_per_crystal = padded_general_wyckoff_ops_mask.long().sum(
        dim=1
    )
    # (B,)

    # Repeat operations for each ASU atom
    padded_general_wyckoff_matrices = torch.repeat_interleave(
        padded_general_wyckoff_matrices,
        n_atoms_per_xtal,
        dim=0,
        output_size=frac_coords.shape[0],
    )
    # (n_asu_atoms_in_batch, 192, 3, 3)
    padded_general_wyckoff_inv_matrices = torch.repeat_interleave(
        padded_general_wyckoff_inv_matrices,
        n_atoms_per_xtal,
        dim=0,
        output_size=frac_coords.shape[0]
    )
    padded_general_wyckoff_translations = torch.repeat_interleave(
        padded_general_wyckoff_translations,
        n_atoms_per_xtal,
        dim=0,
        output_size=frac_coords.shape[0],
    )
    # (n_asu_atoms_in_batch, 192, 1, 3)
    padded_general_wyckoff_ops_mask = torch.repeat_interleave(
        padded_general_wyckoff_ops_mask,
        n_atoms_per_xtal,
        dim=0,
        output_size=frac_coords.shape[0],
    )
    # (n_asu_atoms_in_batch, 192)

    stacked_general_wyckoff_matrices = padded_general_wyckoff_matrices.masked_select(
        padded_general_wyckoff_ops_mask[:, :, None, None].expand(-1, -1, 3, 3)
    ).view(-1, 3, 3)
    # (n_orbited_atoms, 3, 3)
    stacked_general_wyckoff_inv_matrices = padded_general_wyckoff_inv_matrices.masked_select(
        padded_general_wyckoff_ops_mask[:, :, None, None].expand(-1, -1, 3, 3)
    ).view(-1, 3, 3)
    # (n_orbited_atoms, 3, 3)
    stacked_general_wyckoff_translations = padded_general_wyckoff_translations.masked_select(
        padded_general_wyckoff_ops_mask[:, :, None, None].expand(-1, -1, 1, 3)
    ).view(-1, 1, 3)
    # (n_orbited_atoms, 1, 3)

    # - Apply general Wyckoff position operations to orbit asymmetric unit
    #   atoms into the conventional unit cell
    asu_frac_coords_repeated = frac_coords.repeat_interleave(
        general_wyckoff_multiplicity_per_crystal.repeat_interleave(
            n_atoms_per_xtal, dim=0, output_size=frac_coords.shape[0]
        ),
        dim=0,
    )[:, None, :]
    # (n_orbited_atoms, 1, 3)
    conventional_frac_coords_with_dupes = (
        torch.bmm(asu_frac_coords_repeated, stacked_general_wyckoff_matrices)
        + stacked_general_wyckoff_translations
    ) % 1.0
    # (n_orbited_atoms, 1, 3)
    conventional_frac_coords = conventional_frac_coords_with_dupes.view(-1, 3)

    map_atom_to_crystal = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        n_atoms_per_xtal * general_wyckoff_multiplicity_per_crystal,
        dim=0,
        output_size=conventional_frac_coords.shape[0],
    )
    # (n_orbited_atoms,)

    # --
    # De-duplicate identical atoms. Atoms are identical iff they originated from
    # the same ASU atom and overlap.
    # --
    # - We will effectively compute concatenated, flattened adjacency matrices,
    #   where each adjacency matrix is only between atoms of the same orbit.
    orbit_size_per_asu_atom = general_wyckoff_multiplicity_per_crystal.repeat_interleave(
        n_atoms_per_xtal, dim=0, output_size=frac_coords.shape[0]
    )
    # (n_asu_atoms,)
    orbit_size_per_asu_atom_sqr = (orbit_size_per_asu_atom**2).long()
    # (n_asu_atoms,)
    first_asu_atom_index_per_orbit = (
        torch.cumsum(orbit_size_per_asu_atom, dim=0) - orbit_size_per_asu_atom
    )
    # (n_asu_atoms,)
    first_asu_atom_index_per_orbit_expand = torch.repeat_interleave(
        first_asu_atom_index_per_orbit, orbit_size_per_asu_atom_sqr
    )
    # (n_atom_pairs,)
    orbit_size_per_asu_atom_expand = torch.repeat_interleave(
        orbit_size_per_asu_atom,
        orbit_size_per_asu_atom_sqr,
        output_size=first_asu_atom_index_per_orbit_expand.shape[0],
    )
    # (n_atom_pairs,)

    # Compute a tensor containing sequences of numbers that range from 0 to
    # orbit_size_per_asu_atom_sqr for each asu atom. The tensor will be used to
    # compute indices for pairs of atoms. This is a very convoluted way to
    # implement the following (but 10x faster since it removes the for loop):
    # atom_pair_indices = torch.tensor([])
    # for i in range(asu_frac_coords.shape[0]):
    #    atom_pair_indices = torch.cat([atom_pair_indices, torch.arange(orbit_size_per_asu_atom_sqr[i])], dim=0)
    num_atom_pairs = torch.sum(orbit_size_per_asu_atom_sqr)
    index_sqr_offset = (
        torch.cumsum(orbit_size_per_asu_atom_sqr, dim=0) - orbit_size_per_asu_atom_sqr
    ).repeat_interleave(orbit_size_per_asu_atom_sqr)
    # (n_atom_pairs,)
    atom_pair_indices = torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    # (n_atom_pairs,)
    atom_adjacency_row_index = (
        torch.div(atom_pair_indices, orbit_size_per_asu_atom_expand, rounding_mode="floor")
    ) + first_asu_atom_index_per_orbit_expand
    # (n_atom_pairs,)
    atom_adjacency_col_index = (
        atom_pair_indices % orbit_size_per_asu_atom_expand
    ) + first_asu_atom_index_per_orbit_expand
    # (n_atom_pairs,)

    # Ex:
    # asu_frac_coords: [v1, v2, v3]
    # n_asu_coords_per_crystal: [1, 2]
    # general_wyckoff_multiplicity_per_crystal: [3, 2]
    # atom_adjacency_row_index: [0 0 0  1 1 1  2 2 2  3 3  4 4  5 5  6 6]
    # atom_adjacency_col_index: [0 1 2  0 1 2  0 1 2  3 4  3 4  5 6  5 6]
    # map_atom_pair_to_crystal: [0 0 0  0 0 0  0 0 0  1 1  1 1  1 1  1 1]

    # Compute interatomic distances and get mask of unique atom indices
    adjacency_row_atom_coords = conventional_frac_coords[atom_adjacency_row_index]
    adjacency_col_atom_coords = conventional_frac_coords[atom_adjacency_col_index]
    # (n_atom_pairs, 3)

    # # -- The following code block is unstable when fractional coordinates are
    # #    close to 0 or 1, since it does not ensure that differences close to
    # #    -1 or +1 get mapped to zero distances by periodic boundary conditions
    # inter_atom_distances = (adjacency_row_atom_coords - adjacency_col_atom_coords).norm(dim=-1)
    # # (n_atom_pairs,)
    # overlapping_atom_pairs_mask = inter_atom_distances < 1e-6
    # # (n_atom_pairs,)
    # # --
    overlapping_atom_pairs_mask = torch.all(
        torch.abs(
            (adjacency_row_atom_coords - adjacency_col_atom_coords + 0.5) % 1.0 - 0.5
        ) < 1e-6, dim=1
    )
    # (n_atom_pairs,)

    # ---------
    unique_non_overlapping_atom_indices, inverse_indices = torch.unique(
        torch_scatter.segment_coo(
            src=atom_adjacency_col_index[overlapping_atom_pairs_mask],
            index=atom_adjacency_row_index[overlapping_atom_pairs_mask],
            reduce="min",
        ),
        return_inverse=True,
    )
    # (n_unique_orbited_atoms,)
    # ---------

    # De-duplicate atom coordinates. 'unique_non_overlapping_atom_indices'
    # indexes into shape (n_orbited_atoms, *) tensors
    conventional_frac_coords = conventional_frac_coords[unique_non_overlapping_atom_indices]
    # (n_unique_orbited_atoms, 3)
    map_atom_to_crystal = map_atom_to_crystal[unique_non_overlapping_atom_indices]
    # (n_unique_orbited_atoms,)
    num_unit_cell_atoms_per_crystal = torch_scatter.segment_coo(
        src=torch.ones_like(map_atom_to_crystal), index=map_atom_to_crystal, reduce="sum"
    )
    # (B,)
    assert num_unit_cell_atoms_per_crystal.shape[0] == batch_size

    # Get conventional cell Wyckoff and element indices from the ASU
    map_conventional_to_asu_atom_with_dupes = torch.arange(
        frac_coords.shape[0], device=device
    ).repeat_interleave(
        orbit_size_per_asu_atom,
        dim=0,
        output_size=conventional_frac_coords_with_dupes.shape[0],
    )  # (n_orbited_atoms,)
    map_conventional_to_asu_atom = map_conventional_to_asu_atom_with_dupes[
        unique_non_overlapping_atom_indices
    ]
    # (n_unique_orbited_atoms,)
    conventional_wyckoff_indices = wyckoff_indices[map_conventional_to_asu_atom]
    # (n_unique_orbited_atoms,)
    conventional_element_indices = element_indices[map_conventional_to_asu_atom]
    # (n_unique_orbited_atoms,)
    frac_coords_of_conv_atoms = conventional_frac_coords #[map_conventional_to_asu_atom]
    # (n_unique_orbited_atoms, 3)

    # Get general Wyckoff group elements, mod lattice translations
    A_ops = stacked_general_wyckoff_inv_matrices
    # (n_general_wyckoff_ops_mod_lattice_translations, 3, 3)
    t_ops = stacked_general_wyckoff_translations
    # (n_general_wyckoff_ops_mod_lattice_translations, 1, 3)

    return (
        A_ops,
        t_ops,
        inverse_indices,
        map_conventional_to_asu_atom_with_dupes,
        conventional_wyckoff_indices,
        conventional_element_indices,
        frac_coords_of_conv_atoms,
        unique_non_overlapping_atom_indices,
    )


def get_wyckoff_projected_gaussian_noise(
    space_group_indices: Tensor,
    wyckoff_indices: Tensor,
    wyckoff_shape_indices: Tensor,
    n_atoms_per_xtal: Tensor,
    sigma: Union[float, Tensor],
):
    """
    Args:
        space_group_indices: torch.long
            (n_crystals,)
        wyckoff_indices: torch.long
            (n_asu_atoms,)
        wyckoff_shape_indices: torch.long
            (n_asu_atoms,)
        n_atoms_per_xtal: torch.long
            (n_crystals,)
        sigma: Gaussian noise standard deviation
            (,) or (n_asu_atoms, 1)

    Returns:
        shape (n_asu_atoms, 3)
    """
    device = space_group_indices.device

    # Add projected Gaussian noise in fractional space
    unprojected_fractional_noise = sigma * torch.randn(wyckoff_indices.shape[0], 3, device=device)
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
    return projected_fractional_noise
