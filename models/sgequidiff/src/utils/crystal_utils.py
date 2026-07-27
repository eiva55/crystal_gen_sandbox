from collections import Counter
from fractions import Fraction
import json
import cloudpickle

import numpy as np
import torch

from aliases import *
import global_vars


def vmappable_inside(x: Tensor, hull_equations: Tensor, epsilon: float = 1e-6):
    """
    Args:
      x: shape (3,)
      hull_equations: shape (n_linear_shape_bounds, 4)

    Returns:
      shape (,) boolean Tensor
    """
    # (n_linear_shape_bounds, 3) @ (3,) = (n_linear_shape_bounds,)
    return torch.all(
        hull_equations[:, :-1] @ x < -hull_equations[:, -1] - epsilon
    )


is_inside: Callable = torch.vmap(
    vmappable_inside, in_dims=(0, None), out_dims=0
)


def uniformly_sample_point_in_asu_wyckoff_site(
    space_group_numbers: List[str],
    wyckoff_letters: List[str],
    dictionary_of_wyckoffs_in_asu: dict,
    dictionary_of_wyckoff_shape_decompositions: dict,
    device: Union[str, torch.device],
    finished_sampling_mask: tensor = None,
    n_samples_per_wyckoff: int = 1,
    return_sampled_wyckoff_shape_indices: bool = False,
) -> Union[tensor, Tuple[tensor, tensor]]:
    """
    Args:
        space_group_numbers: Length-num_crystals list of space group numbers in
            [1, 230].
        wyckoff_letters: Length-num_crystals list of Wyckoff letters.
        dictionary_of_wyckoffs_in_asu: See
            data/wyckoff_positions/readme.txt
        dictionary_of_wyckoff_shape_decompositions: See
            src/utils/wyckoff_shape_decomp_dict_builder.py
        device:
        finished_sampling_mask: shape (num_crystals,) boolean Tensor which is
            True if this function should not waste computation sampling a point
            for the corresponding crystal.
    Returns:
        (random_samples_in_wyckoffs, sampled_wyckoff_shape_indices) where
            random_samples_in_wyckoffs := shape (n_crystals, n_samples_per_wyckoff, 3)
            sampled_wyckoff_shape_indices := shape (n_crystals, n_samples_per_wyckoff)
    """

    random_samples_in_wyckoffs = []
    sampled_wyckoff_shape_indices = []
    for i, (space_group_number, wyckoff_letter) in enumerate(
        zip(space_group_numbers, wyckoff_letters)
    ):
        wyckoff_position_dict = dictionary_of_wyckoffs_in_asu[space_group_number][wyckoff_letter]
        wyckoff_dof = int(wyckoff_position_dict["dim"])

        if wyckoff_dof == 1:
            # Sample a line segment proportionally to its length
            # 'vertices' is shape (num_line_segments, 2, 3)
            lengths = torch.tensor(
                dictionary_of_wyckoff_shape_decompositions[space_group_number][wyckoff_letter]["volumes"],
                device=device
            )  # (num_line_segments,)
            vertices = torch.tensor(wyckoff_position_dict["vertices"].astype('float32'), device=device)
            # (num_line_segments, 2, 3)
            sampled_line_index = torch.multinomial(
                lengths, num_samples=n_samples_per_wyckoff, replacement=True
            )
            # (n_samples_per_wyckoff,)
            vertices = vertices[sampled_line_index]
            # (n_samples_per_wyckoff, 2, 3)
            sampled_wyckoff_shape_index = sampled_line_index
            # (n_samples_per_wyckoff,)
        elif wyckoff_dof == 2:
            wyckoff_shapes_decomp_dict = dictionary_of_wyckoff_shape_decompositions[
                space_group_number
            ][wyckoff_letter]

            # -- Sample facets proportionally to their areas
            facet_areas = torch.tensor(
                wyckoff_shapes_decomp_dict["volumes"], device=device
            )  # (num_facets,)
            sampled_facet_idxs = torch.multinomial(
                facet_areas, num_samples=n_samples_per_wyckoff, replacement=True
            )
            # (n_samples_per_wyckoff,)

            # -- Sample triangle from each facet according to triangle areas
            max_num_triangles_per_facet: int = wyckoff_shapes_decomp_dict['max_triangles_per_facet']
            _sampled_facet_triangle_areas: List[tensor] = [
                torch.tensor(wyckoff_shapes_decomp_dict['facet_triangle_areas'][facet_index], dtype=torch.float, device=device)
                for facet_index in sampled_facet_idxs
            ]  # length-(n_samples_per_wyckoff) list of (n_triangles,) arrays

            sampled_facet_triangle_areas = torch.zeros((n_samples_per_wyckoff, max_num_triangles_per_facet), device=device)
            for j in range(n_samples_per_wyckoff):
                sampled_facet_triangle_areas[j, :_sampled_facet_triangle_areas[j].shape[0]] = _sampled_facet_triangle_areas[j]
            sampled_triangle_idxs = torch.multinomial(sampled_facet_triangle_areas, num_samples=1).squeeze(dim=1)
            # (n_samples_per_wyckoff,)

            # Index into a (n_facets, max_n_triangles, n_vertices=3, ndim=3)
            # tensor padded with zeroes
            vertices = torch.stack([
                torch.tensor(wyckoff_shapes_decomp_dict['facet_triangles'][facet_index][triangle_index], dtype=torch.float, device=device)
                for facet_index, triangle_index in zip(sampled_facet_idxs, sampled_triangle_idxs)
            ], dim=0)
            # (n_samples_per_wyckoff, n_triangle_vertices=3, ndim=3)
            sampled_wyckoff_shape_index = sampled_facet_idxs
            # (n_samples_per_wyckoff,)
        else:
            vertices = wyckoff_position_dict["vertices_tensor"].to(device)
            sampled_wyckoff_shape_index = torch.tensor(
                [0], dtype=torch.long, device=device
            ).expand(n_samples_per_wyckoff)

        if finished_sampling_mask is not None and finished_sampling_mask[i]:
            sample_in_wyckoff = torch.tensor(
                [[-1.0, -1.0, -1.0]], dtype=torch.float, device=device
            ).expand(n_samples_per_wyckoff, 3)
            sampled_wyckoff_shape_index = torch.tensor(
                [-1], dtype=torch.long, device=device
            ).expand(n_samples_per_wyckoff)
        else:
            sample_in_wyckoff = uniformly_sample_point_in_convex_shape(
                space_group_number=int(space_group_number),
                vertices=vertices,
                wyckoff_site_dimensionality=wyckoff_dof,
                device=device,
                n_samples=n_samples_per_wyckoff,
            )  # (n_samples_per_wyckoff, 3)
            assert sample_in_wyckoff is not None
        random_samples_in_wyckoffs.append(sample_in_wyckoff)
        sampled_wyckoff_shape_indices.append(sampled_wyckoff_shape_index)
    random_samples_in_wyckoffs = torch.stack(random_samples_in_wyckoffs, dim=0)
    sampled_wyckoff_shape_indices = torch.stack(sampled_wyckoff_shape_indices, dim=0)
    if return_sampled_wyckoff_shape_indices:
        return random_samples_in_wyckoffs, sampled_wyckoff_shape_indices
        # (n_crystals, n_samples_per_wyckoff, 3), (n_crystals, n_samples_per_wyckoff)
    else:
        return random_samples_in_wyckoffs  # (n_crystals, n_samples_per_wyckoff, 3)


def uniformly_sample_point_in_convex_shape(
    space_group_number: int,
    vertices: tensor,
    wyckoff_site_dimensionality: int,
    device: Union[torch.device, str],
    n_samples: int = 1,
) -> tensor:
    """
    Given list of convex shape vertices, return a random point inside the
    convex shape. The Wyckoff site can be any of the following:
        0D point: shape (1, 3) tensor
        1D line segment: shape (n_samples, n_segment_vertices=2, 3) tensor
        2D triangle: shape (n_samples, n_triangle_vertices=3, ndim=3) tensor
        3D polytope: shape (n_vertices, 3) tensor

    Args:
        space_group_number: Must be in [1, 230]. Only used if
            'wyckoff_site_dimensionality' is in [2, 3].
        vertices: See above
        wyckoff_site_dimensionality: Must be in [0, 1, 2, 3]
        device:
        n_samples: Number of points to sample in the convex shape(s)

    Returns:
        shape (n_samples, 3) tensor
    """
    if wyckoff_site_dimensionality == 0:
        assert vertices.shape == (1, 3)
        return vertices.expand(n_samples, 3)
        # (n_samples, 3)
    elif wyckoff_site_dimensionality == 1:
        assert vertices.shape == (n_samples, 2, 3)
        samples = torch.rand((n_samples, 1), device=device)  # (n_samples, 1)
        end_point1, end_point2 = (
            vertices[:, 0],
            vertices[:, 1],
        )  # (n_samples, 3), (n_samples, 3)
        samples = samples * (end_point2 - end_point1) + end_point1
        return samples  # (n_samples, 3)
    elif wyckoff_site_dimensionality == 2:
        assert vertices.shape == (n_samples, 3, 3)
        # See Section 4.2 for algo to sample uniformly from triangle:
        # https://www.cs.princeton.edu/~funk/tog02.pdf
        #   P = (1-sqrt(r1)) * A + sqrt(r1) * (1-r2) * B + sqrt(r1) * r2 * C
        # where triangle has vertices (A, B, C) and r1 and r2 ~ Unif[0, 1]
        r1_sqrt = torch.rand((n_samples, 1), device=device).sqrt()
        r2 = torch.rand((n_samples, 1), device=device)
        samples = (
            (1.0 - r1_sqrt) * vertices[:, 0, :]
            + r1_sqrt * (1.0 - r2) * vertices[:, 1, :]
            + r1_sqrt * r2 * vertices[:, 2, :]
        )
        return samples  # (n_samples, 3)
    elif wyckoff_site_dimensionality == 3:
        # 'vertices' is (n_vertices, 3) tensor
        box_lower_left, _ = torch.min(vertices, dim=0)  # [xmin, ymin, zmin]
        box_top_right, _ = torch.max(vertices, dim=0)  # [xmax, ymax, zmax]
        asu_hull_equations = global_vars.asu_hull_equations[space_group_number - 1].to(device)
        # (n_linear_shape_bounds, 4)

        accepted_samples = []
        total_n_accepted = 0
        while True:
            samples = torch.rand(3 * n_samples, 3, device=device)
            samples = (
                samples * (box_top_right - box_lower_left) + box_lower_left
            )  # (3 * n_samples, 3)
            samples_are_in_asu = is_inside(samples, asu_hull_equations)  # (3 * n_samples,)
            accepted_samples.append(samples[samples_are_in_asu])
            total_n_accepted += samples_are_in_asu.long().sum()
            if total_n_accepted >= n_samples:
                return torch.cat(accepted_samples, dim=0)[:n_samples]
    else:
        raise AttributeError
