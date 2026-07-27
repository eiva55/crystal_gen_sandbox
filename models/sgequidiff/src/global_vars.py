from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pyxtal.symmetry import Group as PyxtalGroup
from scipy.spatial import ConvexHull

from utils.io_utils import get_project_dir, DATA_DIRECTORY
from aliases import *
from spacegroup_data import spgroup_data
from constants import MAX_WYCKOFF_SITES


def load_dictionary_of_wyckoff_sites_in_asus(
    json_filepath: str = Path(
        DATA_DIRECTORY / "wyckoff_positions/clean_wyckoffs_in_asu_v6.json"
    ).as_posix(),
) -> dict:
    """
    See 'data/wyckoff_positions/readme.txt'.
    """

    with open(json_filepath) as file:
        wyckoffs_dict = json.load(file)

    # Convert json described in data/wyckoff_positions/readme.txt to a
    # dictionary with 'vertices' values as numpy arrays of Fraction objects
    # instead of as nested lists of strings and 'plane_coefficients' values as
    # nested lists of Fraction objects instead of strings.
    # Also convert 'dim' values from strings to integers.
    for space_group_number in wyckoffs_dict.keys():
        for wyckoff_letter in wyckoffs_dict[space_group_number]["ordered_wyckoff_letters"]:
            wyckoff_site_dict = wyckoffs_dict[space_group_number][wyckoff_letter]
            wyckoff_dof = int(wyckoff_site_dict["dim"])

            convert_string_array_to_fractions = np.vectorize(string_to_fraction)
            if wyckoff_dof != 2:
                wyckoff_site_dict["vertices"] = convert_string_array_to_fractions(
                    wyckoff_site_dict["vertices"]
                )
                wyckoff_site_dict["vertices_tensor"] = torch.tensor(
                    wyckoff_site_dict["vertices"].astype("float32"),
                    dtype=torch.float,
                )
            else:
                all_faces: List[ndarray] = []
                all_faces_tensors: List[tensor] = []
                for face in wyckoff_site_dict["vertices"]:
                    face_array = convert_string_array_to_fractions(
                        face
                    )  # (n_face_vertices, 3)
                    all_faces.append(face_array)
                    all_faces_tensors.append(
                        torch.tensor(
                            face_array.astype("float32"), dtype=torch.float
                        )
                    )
                wyckoff_site_dict["vertices"] = all_faces
                wyckoff_site_dict["vertices_tensors"] = all_faces_tensors
            wyckoff_site_dict["dim"] = int(wyckoff_site_dict["dim"])

            if wyckoff_site_dict["dim"] == 2:
                wyckoff_site_dict["plane_coefficients"] = convert_string_array_to_fractions(
                    wyckoff_site_dict["plane_coefficients"]
                ).tolist()

    return wyckoffs_dict


def string_to_fraction(string: str) -> Fraction:
    return Fraction(string)


# GLOBAL VARIABLES TODO deprecate this and add to IO_UTILS
PROJECT_DIR = get_project_dir()
embedding_tools = None  # see src.utils.embedding_utils.EmbeddingTools
ASU_DICT_PATH = Path(
    DATA_DIRECTORY / "wyckoff_positions/clean_wyckoffs_in_asu_v6.json"
).as_posix()
asu_wyckoff_dict = load_dictionary_of_wyckoff_sites_in_asus(ASU_DICT_PATH)

# Pyxtal space group construction is slow, so pre-compute them.
# Converting every Wyckoff SymmOp to torch.Tensor objects on-the-fly is also
# a bottleneck in graph construction, so pre-compute the conversions.
pyxtal_space_group_dict = {}
padded_general_wyckoff_matrices = torch.zeros(230, 192, 3, 3)
padded_inverse_general_wyckoff_matrices = torch.zeros(230, 192, 3, 3)
padded_general_wyckoff_translations = torch.zeros(230, 192, 1, 3)
padded_general_wyckoff_ops_mask = torch.zeros(230, 192, dtype=torch.bool)
# Mask will be False for entries that correspond to padding
for space_group_number in range(1, 231):
    pyxtal_space_group = PyxtalGroup(space_group_number, style="pyxtal")
    for i, wyckoff_position in enumerate(pyxtal_space_group.Wyckoff_positions):
        if i == 0:
            # Convert general Wyckoff positions ops to tensors
            assert wyckoff_position.get_dof() == 3

            tensor_wyckoff_rotations = []
            tensor_wyckoff_inv_rotations = []  # for non-trigonal/hexagonal xtals, these will simply be rotations transposed
            tensor_wyckoff_translations = []
            for symmetry_rep in wyckoff_position.ops:
                rotation = torch.tensor(symmetry_rep.rotation_matrix, dtype=torch.float32)
                tensor_wyckoff_rotations.append(rotation.T)  # (3, 3)
                tensor_wyckoff_inv_rotations.append(torch.linalg.inv(rotation).T)
                tensor_wyckoff_translations.append(
                    torch.tensor(
                        symmetry_rep.translation_vector, dtype=torch.float32
                    ).unsqueeze(0)
                )  # (1, 3)

            tensor_wyckoff_rotations = torch.stack(tensor_wyckoff_rotations, dim=0)
            # (general_wyckoff_position.multiplicity, 3, 3)
            tensor_wyckoff_inv_rotations = torch.stack(tensor_wyckoff_inv_rotations, dim=0)
            # (general_wyckoff_position.multiplicity, 3, 3)
            tensor_wyckoff_translations = torch.stack(tensor_wyckoff_translations, dim=0)
            # (general_wyckoff_position.multiplicity, 1, 3)
            wyckoff_position.ops = {
                "rotations": tensor_wyckoff_rotations,
                "inv_rotations": tensor_wyckoff_inv_rotations,
                "translations": tensor_wyckoff_translations,
            }
            """
            Example of how to use these transformations:
            
            asu_coord = torch.rand(3)
            rotations = wyckoff_position.ops['rotations']
            translations = wyckoff_position.ops['translations']
            orbit = (
                asu_coord[None, None, :] @ rotations + translations
            ).view(-1, 3) % 1.0
            # orbit.shape: (general_wyckoff_position.multiplicity, 3)
            """

            # For batching the conversion of ASU fractional coordinates to
            # primitive cell cartesian coordinates, store all general Wyckoff
            # position symmetry operations into a shape
            # (n_space_groups=230, max_operations=192, 3, 3) padded tensor of
            # (3, 3) matrices and a shape
            # (n_space_groups=230, max_operations=192, 1, 3) tensor of (1, 3)
            # translations.
            padded_general_wyckoff_matrices[space_group_number - 1][
                : tensor_wyckoff_rotations.shape[0]
            ] = tensor_wyckoff_rotations
            padded_inverse_general_wyckoff_matrices[space_group_number - 1][
                : tensor_wyckoff_rotations.shape[0]
            ] = tensor_wyckoff_inv_rotations
            padded_general_wyckoff_translations[space_group_number - 1][
                : tensor_wyckoff_translations.shape[0]
            ] = tensor_wyckoff_translations
            padded_general_wyckoff_ops_mask[space_group_number - 1][
                : tensor_wyckoff_rotations.shape[0]
            ] = True

            # Assert first operation is always the identity
            assert (
                torch.equal(
                    padded_general_wyckoff_matrices[space_group_number - 1, 0],
                    torch.eye(3)
                )
                and
                torch.equal(
                    padded_general_wyckoff_translations[space_group_number - 1, 0],
                    torch.zeros(1, 3)
                )
                and
                padded_general_wyckoff_ops_mask[space_group_number - 1][0] == True
            )
        else:
            # Delete special Wyckoff ops to save memory
            assert wyckoff_position.get_dof() < 3
            wyckoff_position.ops = None

    pyxtal_space_group_dict[space_group_number] = pyxtal_space_group

# --- On-the-fly tensor construction is slow, so pre-compute them when possible
# - P matrices
conventional_to_primitive_transforms: dict = {
    "cP": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
    },
    "tP": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
    },
    "hP": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
    },
    "oP": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
    },
    "mP": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
    },
    "cF": {
        "P": torch.tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.0, -0.5], [0.0, -0.5, -0.5]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]], dtype=torch.float32
        ),
    },
    "oF": {
        "P": torch.tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.0, -0.5], [0.0, -0.5, -0.5]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]], dtype=torch.float32
        ),
    },
    "cI": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float32
        ),
    },
    "tI": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float32
        ),
    },
    "oI": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float32
        ),
    },
    "hR": {
        "P": (1.0 / 3.0)
        * torch.tensor(
            [[-3.0, -3.0, 0.0], [-3.0, 0.0, 0.0], [-2.0, -1.0, -1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -3.0]], dtype=torch.float32
        ),
    },
    "oC": {
        "P": torch.tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32
        ),
    },
    "oA": {
        "P": torch.tensor(
            [[0.0, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.5, -0.5]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]], dtype=torch.float32
        ),
    },
    "mC": {
        "P": torch.tensor(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.5, -0.5, 0.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, -1.0, -2.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        ),
    },
    "aP": {
        "P": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        "invP": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
    },
}

# Construct (230, 3, 3) tensors for advanced indexing
conventional_to_primitive_P_matrices = []
conventional_to_primitive_invP_matrices = []
for space_group_number in range(1, 231):
    bravais_lattice_string = spgroup_data[space_group_number][0]
    conventional_to_primitive_P_matrices.append(
        conventional_to_primitive_transforms[bravais_lattice_string]["P"]
    )
    conventional_to_primitive_invP_matrices.append(
        conventional_to_primitive_transforms[bravais_lattice_string]["invP"]
    )
conventional_to_primitive_P_matrices = torch.stack(conventional_to_primitive_P_matrices, dim=0)
conventional_to_primitive_invP_matrices = torch.stack(
    conventional_to_primitive_invP_matrices, dim=0
)
# ---

# -- Compute a (230, MAX_WYCKOFF_SITES) torch.long tensor of Wyckoff
#   dimensionalities for batched indexing
wyckoff_dimension_tensor: Tensor = -1 * torch.ones(230, 27, dtype=torch.long)
for space_group_number in range(1, 231):
    sorted_wyckoff_letters = asu_wyckoff_dict[str(space_group_number)][
        "ordered_wyckoff_letters"
    ]
    # sorted alphabetically
    for wyckoff_index, wyckoff_letter in enumerate(sorted_wyckoff_letters):
        wyckoff_dim = asu_wyckoff_dict[str(space_group_number)][wyckoff_letter]["dim"]
        if wyckoff_dim == 0:
            wyckoff_dimension_tensor[space_group_number - 1][wyckoff_index] = 0
        elif wyckoff_dim == 1:
            wyckoff_dimension_tensor[space_group_number - 1][wyckoff_index] = 1
        elif wyckoff_dim == 2:
            wyckoff_dimension_tensor[space_group_number - 1][wyckoff_index] = 2
        elif wyckoff_dim == 3:
            wyckoff_dimension_tensor[space_group_number - 1][wyckoff_index] = 3

# -- Compute (230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff, 3, 3) tensor
# holding projection matrices. They project noise vectors onto points, lines,
# planes, and 3D space associated with each convex Wyckoff shape in the ASU.
# Simultaneously compute (230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff) tensor
# holding Wyckoff shape volumes.
def _project_onto_1d_subspace(line: Tensor):
    """
    Args:
        line: shape (1, 3) tensor
    Returns:
        projection_matrix: (3, 3). E.g., for x with shape (1, 3),
            projected_x = x @ projection_matrix
    """
    projection_matrix = (line.T @ line) / (line ** 2).sum()
    return projection_matrix


def _project_onto_2d_subspace(facet_vertices: Tensor, return_plane_normal: bool = False):
    """
    Args:
        facet_vertices: (n_vertices, 3)
    Returns:
        projection_matrix: (3, 3). E.g., for x with shape (1, 3),
            projected_x = x @ projection_matrix
    """
    ab = facet_vertices[0] - facet_vertices[1]
    bc = facet_vertices[1] - facet_vertices[2]
    plane_normal = torch.linalg.cross(ab, bc)

    # get orthogonal inplane vectors
    v1 = torch.linalg.cross(ab, plane_normal).view(1, 3)
    v2 = ab.view(1, 3)

    projection_matrix = ((v1.T @ v1) / (v1 ** 2).sum()) + ((v2.T @ v2) / (v2 ** 2).sum())
    if return_plane_normal:
        return projection_matrix, plane_normal
    else:
        return projection_matrix

# num_wyckoffs = 1731
num_space_groups = 230
max_wyckoffs_per_space_group = 27
max_shapes_per_wyckoff = 4

noise_projection_matrices = torch.zeros(
    (num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3, 3)
)  # (num_space_groups, max_wyckoffs_per_space_group, max_num_shapes, 3, 3)
wyckoff_shape_volumes = torch.zeros(
    (num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff)
)  # (num_space_groups, max_wyckoffs_per_space_group, max_num_shapes)
point_per_1d_wyckoff_line = torch.zeros(
     (num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3)
)
line_directions_of_1d_wyckoffs = torch.zeros(
    (num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3)
)
point_per_2d_wyckoff_plane = torch.zeros(
    (num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3)
)
plane_normals_of_2d_wyckoffs = torch.zeros(
    (num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3)
)
for sg_num in range(1, 231):
    sg_dict = asu_wyckoff_dict[str(sg_num)]
    wyckoff_projection_matrices = torch.zeros((max_shapes_per_wyckoff, 3, 3))
    for i, wyckoff_letter in enumerate(sg_dict["ordered_wyckoff_letters"]):
        wyckoff_dict = sg_dict[wyckoff_letter]

        # store wyckoff shape volumes in tensor
        shape_volumes: List[float] = wyckoff_dict["volumes"]
        wyckoff_shape_volumes[sg_num - 1, i, :len(shape_volumes)] = (
            torch.tensor(shape_volumes, dtype=torch.float)
        )

        dim = int(wyckoff_dict["dim"])
        if dim == 0:
            wyckoff_projection_matrices[0] = torch.zeros(3, 3)
        elif dim == 1:
            for j, line_segment in enumerate(wyckoff_dict["vertices"]):
                line_segment = line_segment.astype("float32")
                line = torch.tensor(line_segment[1] - line_segment[0]).view(1, 3)
                projection_matrix = _project_onto_1d_subspace(line)  # (3, 3)
                wyckoff_projection_matrices[j] = projection_matrix

                point_per_1d_wyckoff_line[sg_num - 1, i, j] = torch.tensor(line_segment[1]).view(3)
                line_directions_of_1d_wyckoffs[sg_num - 1, i, j] = line.view(3)
        elif dim == 2:
            for j, facet_vertices in enumerate(wyckoff_dict["vertices"]):
                projection_matrix, plane_normal = _project_onto_2d_subspace(
                    torch.tensor(facet_vertices.astype("float32")),
                    return_plane_normal=True,
                )
                wyckoff_projection_matrices[j] = projection_matrix

                point_per_2d_wyckoff_plane[sg_num - 1, i, j] = torch.tensor(
                    facet_vertices.astype("float32")[0]
                )
                plane_normals_of_2d_wyckoffs[sg_num - 1, i, j] = plane_normal
        elif dim == 3:
            wyckoff_projection_matrices[0] = torch.eye(3)
        else:
            raise AttributeError
        noise_projection_matrices[sg_num - 1][i] = wyckoff_projection_matrices
# ---

# -- Compute (230, max simplicial hull facets per asu, 4) tensor of padded
# equations to determine whether a point is inside the ASU, and a boolean mask
# of shape (230, max simplicial hull facets per asu) which is False to indicate
# padding
max_simplicial_hull_facets = 16  # found empirically
# Initialize hull equations with padding that makes the inside test
# always evaluate to True
asu_hull_equations = -1.0 * nn.functional.one_hot(
    torch.tensor(3), num_classes=4
)[None, None, :].repeat(230, max_simplicial_hull_facets, 1)
# (n_space_groups, n_linear_shape_bounds, 4)
asu_hull_equations_mask = torch.zeros(230, max_simplicial_hull_facets, dtype=torch.bool)
for sg in range(1, 231):
    sg_dict = asu_wyckoff_dict[str(sg)]
    general_wyckoff_letter = sg_dict["ordered_wyckoff_letters"][-1]
    general_wyckoff_dict = sg_dict[general_wyckoff_letter]

    assert general_wyckoff_dict["dim"] == 3
    asu_vertices: ndarray = general_wyckoff_dict["vertices"]
    # (n_vertices, 3)

    hull = ConvexHull(asu_vertices.astype("float64"))
    equations = hull.equations
    # (n_facet, ndim+1)
    n_simplicial_facets = equations.shape[0]

    asu_hull_equations[sg - 1, :n_simplicial_facets] = torch.tensor(equations, dtype=torch.float)
    asu_hull_equations_mask[sg - 1, :n_simplicial_facets] = True
asu_hull_equations_numpy = asu_hull_equations.numpy()

# -- Compute:
# wyckoff_shape_vertices: torch.float
#   (230, MAX_WYCKOFF_SITES, max shapes per Wyckoff, max vertices per shape, 3)
#   shape padded tensor of Wyckoff shape vertices in the ASU.
# mask_wyckoff_shape_vertices: torch.bool
#   (230, MAX_WYCKOFF_SITES, max shapes per Wyckoff, max vertices per shape)
#   shape tensor which is False for padding and True for valid shape vertices.
# n_vertices_per_wyckoff_shape: torch.long
#   (230, MAX_WYCKOFF_SITES, max shapes per Wyckoff) shape tensor
# n_shapes_per_wyckoff: torch.long
#   (230, MAX_WYCKOFF_SITES) shape tensor
max_vertices_per_wyckoff_shape = 10  # 10 if including 3D Wyckoffs. Else 5.

wyckoff_shape_vertices = torch.zeros(
    230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff, max_vertices_per_wyckoff_shape, 3,
    dtype=torch.float,
)
mask_wyckoff_shape_vertices = torch.zeros(
    230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff, max_vertices_per_wyckoff_shape,
    dtype=torch.bool,
)
n_vertices_per_wyckoff_shape = torch.zeros(
    230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff, dtype=torch.long
)
n_shapes_per_wyckoff = torch.zeros(230, MAX_WYCKOFF_SITES, dtype=torch.long)
for sg in range(1, 231):
    sg_dict = asu_wyckoff_dict[str(sg)]
    for wp_idx, wyckoff_letter in zip(
        range(MAX_WYCKOFF_SITES), sg_dict["ordered_wyckoff_letters"]
    ):
        wyck_dict = sg_dict[wyckoff_letter]
        wyck_dof = int(wyck_dict["dim"])
        if not wyck_dof in [1, 2]:
            # only 1 shape
            n_vertices = wyck_dict["vertices_tensor"].shape[0]
            n_vertices_per_wyckoff_shape[sg-1, wp_idx, 0] = n_vertices
            n_shapes_per_wyckoff[sg-1, wp_idx] = 1
            wyckoff_shape_vertices[sg-1, wp_idx, 0, :wyck_dict["vertices"].shape[0], :] = (
                wyck_dict["vertices_tensor"]
            )
            # (n_vertices, 3)
            mask_wyckoff_shape_vertices[sg-1, wp_idx, 0, :n_vertices] = True
        else:
            n_shapes = len(wyck_dict["vertices"])  # number of segments or facets
            n_shapes_per_wyckoff[sg-1, wp_idx] = n_shapes
            assert 0 <= n_shapes <= max_shapes_per_wyckoff
            if wyck_dof == 1:
                n_vertices = 2
                n_vertices_per_wyckoff_shape[sg-1, wp_idx, :n_shapes] = n_vertices
                wyckoff_shape_vertices[sg-1, wp_idx, :n_shapes, :n_vertices, :] = (
                    wyck_dict["vertices_tensor"]
                )
                # (n_segments, 2, 3)
                mask_wyckoff_shape_vertices[sg-1, wp_idx, :n_shapes, :n_vertices] = True
            elif wyck_dof == 2:
                # wyck_dict["vertices_tensors"]: length-(n_faces) List[Tensor]
                # of shape (n_face_vertices, 3) tensors
                for shape_idx in range(n_shapes):
                    n_vertices = len(wyck_dict["vertices"][shape_idx])
                    n_vertices_per_wyckoff_shape[sg-1, wp_idx, shape_idx] = n_vertices
                    wyckoff_shape_vertices[sg-1, wp_idx, shape_idx, :n_vertices, :] = (
                        wyck_dict["vertices_tensors"][shape_idx]
                    )
                    # (n_vertices, 3)
                    mask_wyckoff_shape_vertices[sg-1, wp_idx, shape_idx, :n_vertices] = True
