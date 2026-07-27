from collections import Counter
from fractions import Fraction
from random import random
import warnings

import numpy as np
from torch_scatter import scatter, segment_coo, segment_csr, scatter_min
from pymatgen.core import Lattice, Structure

from spacegroup_data import spgroup_data
from constants import *
from aliases import *
from utils.io_utils import get_random_hash
from crystal_classes import ASUCrystal
import global_vars


def stack_ragged(ndarrays: List[ndarray], axis: Optional[int] = 0) -> Tuple[ndarray, ndarray]:
    lengths: List[int] = [np.shape(a)[axis] for a in ndarrays]
    indices: ndarray = np.cumsum(lengths[:-1])
    stacked: ndarray = np.concatenate(ndarrays, axis=axis)
    return stacked, indices


def pack_crystals(crystals: List[ASUCrystal]) -> Tuple[ndarray, ndarray]:
    flat_crystals: List[ndarray] = [crystal.flatten() for crystal in crystals]
    return stack_ragged(flat_crystals)


def pack_and_save(crystals: List[ASUCrystal], save_path: Path) -> None:
    packed, indices = pack_crystals(crystals)
    np.savez(save_path, packed=packed, indices=indices)


def unpack(path: Path) -> List[ASUCrystal]:
    npz: dict = np.load(path)
    indices: ndarray = npz["indices"]
    packed: ndarray = npz["packed"]
    crystals = [
        ASUCrystal.from_flat(flat_crystal) for flat_crystal in np.split(packed, indices)
    ]
    return crystals


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.
    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.
    If a global seed is set, the augmentations are deterministic.
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def frac_to_cart_coords(
    frac_coords,
    num_atoms_per_crystal,
    lattice_lengths=None,
    lattice_angles=None,
    lattice_matrix=None,
):
    """
    Definitions:
    n_crystals := number of crystals in the batch
    n_atoms := total number of atoms in the batch

    Args:
        frac_coords: torch.tensor of shape (n_atoms, 3), fractional coordinates
        lattice_lengths: torch.tensor of shape (n_crystals, 3), Angstroms
        lattice_angles: torch.tensor of shape (n_crystals, 3), degrees
        num_atoms_per_crystal: torch.LongTensor of shape (n_crystals,), int.
            Provides a tensor containing the number of atoms per crystal in
            'frac_coords'.
        lattice_matrix: torch.tensor of shape (n_crystals, 3, 3). For each 3x3
            matrix, each row is a lattice vector.

    Returns:
        cart_coords: torch.tensor of shape (n_atoms, 3), Cartesian coordinates
    """
    if lattice_matrix is None:
        assert None not in [lattice_lengths, lattice_angles]
        lattice_matrix = lattice_params_to_matrix_torch(lattice_lengths, lattice_angles)
    lattice_nodes = torch.repeat_interleave(
        lattice_matrix, num_atoms_per_crystal, dim=0
    ).float()
    cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

    return cart_coords


def cart_to_frac_coords(
    cart_coords,
    num_atoms_per_crystal,
    lattice_lengths=None,
    lattice_angles=None,
    lattice_matrix=None,
    mod_lattice_translations: bool = True,
):
    """
    See 'frac_to_cart_coords' docstring above.
    """
    if lattice_matrix is None:
        assert lattice_lengths is not None and lattice_angles is not None
        lattice_matrix = lattice_params_to_matrix_torch(lattice_lengths, lattice_angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice_matrix)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms_per_crystal, dim=0)
    frac_coords = torch.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    if mod_lattice_translations:
        frac_coords = frac_coords % 1.0
    return frac_coords


def lattice_params_to_matrix_torch(lattice_lengths, lattice_angles):
    """
    Torchified version of pymatgen.core.lattice.from_parameters().
    Batched torch version to compute lattice matrix from params.

    N := number of crystals in the batch

    Args:
        lattice_lengths: torch.Tensor of shape (N, 3), unit A
        lattice_angles: torch.Tensor of shape (N, 3), unit degree

    Returns:
        torch.tensor of shape (N, 3, 3)
    """
    angles_r = torch.deg2rad(lattice_angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lattice_lengths[:, 0] * sins[:, 1],
            torch.zeros(lattice_lengths.size(0), device=lattice_lengths.device),
            lattice_lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lattice_lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lattice_lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lattice_lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lattice_lengths.size(0), device=lattice_lengths.device),
            torch.zeros(lattice_lengths.size(0), device=lattice_lengths.device),
            lattice_lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def vmappable_lattice_matrix_to_params(matrix: Tensor):
    """
    Modeled after pymatgen.core.Lattice.lengths and pymatgen.core.Lattice.angles
    Args:
        matrix: torch.float
            shape (3, 3)
    Returns:
        lengths: torch.float
            shape (3,)
        angles: torch.float
            shape (3,)
    """
    lengths = torch.sqrt(torch.sum(matrix**2, dim=1))

    j = torch.tensor([1, 2, 0], dtype=torch.long, device=matrix.device)
    k = torch.tensor([2, 0, 1], dtype=torch.long, device=matrix.device)
    angles = torch.clamp(torch.sum(matrix[j] * matrix[k], dim=1) / (lengths[j] * lengths[k]), -1.0, 1.0)
    angles = torch.arccos(angles) * 180.0 / torch.pi
    return lengths, angles


lattice_matrix_to_params: Callable = torch.vmap(
    vmappable_lattice_matrix_to_params, in_dims=0, out_dims=0
)


def get_P_matrix(
    bravais_lattice: str, device: torch.device = torch.device("cpu"), dtype=torch.float32
):
    """
    Return a tuple of length 2 with the P matrix and its inverse::
      (P, invP)
    with :math:`invP = P^{-1}`.
    These :math:`P` matrices are obtained from Table 3 of the HPKOT
    paper.
    The P matrix is a :math:`3\times 3` matrix is the matrix that converts
    the lattice vectors from crystallographic conventional
    :math:`(a,b,c)` to crystallographic primitive :math:`(a_P, b_P, c_P)`
    as follows: :math:`(a_P, b_P, c_P) = (a,b,c) P`
    The change of (real space) coordinate triples follows instead:
    :math:`(x_P, y_P, z_P)^T = (P^{-1}) (x,y,z)^T`
    .. note:: the :math:`invP = P^{-1}` matrix is always integer (with values
        only :math:`-1, 0, 1`) while :math:`P` is rational (non-integer values can be
        :math:`\pm \frac 1 2` and :math:`\pm \frac 1 3`).
    """

    if bravais_lattice in ["cP", "tP", "hP", "oP", "mP"]:
        P = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        invP = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
    elif bravais_lattice in ["cF", "oF"]:
        P = torch.tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.0, -0.5], [0.0, -0.5, -0.5]],
            device=device,
            dtype=dtype,
        )
        invP = torch.tensor(
            [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
            device=device,
            dtype=dtype,
        )
    elif bravais_lattice in ["cI", "tI", "oI"]:
        P = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], device=device, dtype=dtype
        )
        invP = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], device=device, dtype=dtype
        )
    elif bravais_lattice == "hR":  # todo: SEEKPATH DOES NOT MATCH SPGLIB
        # spglib
        P = (
            1.0
            / 3.0
            * torch.tensor(
                [[-3.0, -3.0, 0.0], [-3.0, 0.0, 0.0], [-2.0, -1.0, -1.0]],
                device=device,
                dtype=dtype,
            )
        )
        invP = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -3.0]], device=device, dtype=dtype
        )

        # seekpath
        # P = 1.0 / 3.0 * torch.tensor(
        #     [[2., -1., -1.],
        #      [1., 1., -2.],
        #      [1., 1., 1.]])
        # invP = torch.tensor(
        #     [[1., 0., 1.],
        #      [-1., 1., 1.],
        #      [0., -1., 1.]])
    elif bravais_lattice == "oC":
        # spglib
        P = torch.tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, -1.0]], device=device, dtype=dtype
        )
        invP = torch.tensor(
            [[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, -1.0]], device=device, dtype=dtype
        )

        # seekpath
        # P = 1.0 / 2.0 * torch.tensor(
        #     [[1., 1., 0.],
        #      [-1., 1., 0.],
        #      [0., 0., 2.]])
        # invP = torch.tensor(
        #     [[1., -1., 0.],
        #      [1., 1., 0.],
        #      [0., 0., 1.]])
    elif bravais_lattice == "oA":
        # spglib
        P = torch.tensor(
            [[0.0, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.5, -0.5]], device=device, dtype=dtype
        )
        invP = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype
        )

        # seekpath
        # P = 1.0 / 2.0 * torch.tensor(
        #     [[0., 0., 2.],
        #      [1., 1., 0.],
        #      [-1., 1., 0.]])
        # invP = torch.tensor(
        #     [[0., 1., -1.],
        #      [0., 1., 1.],
        #      [1., 0., 0.]])
    elif bravais_lattice == "mC":
        # spglib
        P = torch.tensor(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.5, -0.5, 0.0]], device=device, dtype=dtype
        )
        invP = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, -1.0, -2.0], [1.0, 0.0, 0.0]], device=device, dtype=dtype
        )

        # seekpath
        # P = 1.0 / 2.0 * torch.tensor(
        #     [[1., -1., 0.],
        #      [1., 1., 0.],
        #      [0., 0., 2.]])
        # invP = torch.tensor(
        #     [[1., 1., 0.],
        #      [-1., 1., 0.],
        #      [0., 0., 1.]])
    elif bravais_lattice == "aP":
        # For aP, I should have already obtained the primitive cell
        P = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
        invP = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
    else:
        raise ValueError("Invalid bravais_lattice {}".format(bravais_lattice))

    return P, invP


def torch_legal_lattice_parameters(
    space_groups: tensor,
    lattice_parameters: tensor,
    lattice_log_probs: tensor,
    device: Union[torch.device, str],
) -> Tuple[tensor, tensor, tensor]:
    """
    Enforce constraints on lattice vectors given the space group.
    Take in 3 idpt lattice angles and lengths. Return lattice angles and
    lengths which satisfy the space group.

    Zero out logPf corresponding to elements in 'lattice_parameters' which end
    up being unused.

    Args:
        space_groups: shape (batch_size,)
        lattice_parameters: shape (batch_size, 6) where the
            first 3 columns correspond to lengths (a, b, c) and the last 3
            correspond to the angles (alpha, beta, gamma)
        lattice_log_probs: shape (batch_size, 6). Log probability of
            sampling each lattice parameter.

    Returns:
        (lattice_angles, lattice_lengths, masked_lattice_log_probs)
        Shapes: (batch_size, 3), (batch_size, 3), (batch_size, 6)
    """
    lattice_lengths, lattice_angles = torch.chunk(lattice_parameters, 2, dim=1)

    length_projection_matrices = []
    angle_projection_matrices, angle_translation_vectors = [], []
    log_prob_masks = []
    for spacegroup in space_groups:
        assert 1 <= spacegroup <= 230
        (
            length_matrix,
            angle_matrix,
            angle_vector,
            log_prob_mask,
        ) = lattice_transform_and_log_prob_mask(spacegroup, device)
        # (3, 3), (3, 3), (3,), (6,)

        length_projection_matrices.append(length_matrix)
        angle_projection_matrices.append(angle_matrix)
        angle_translation_vectors.append(angle_vector)
        log_prob_masks.append(log_prob_mask)

    length_projection_matrices = torch.stack(length_projection_matrices, dim=0)
    angle_projection_matrices = torch.stack(angle_projection_matrices, dim=0)
    angle_translation_vectors = torch.stack(angle_translation_vectors, dim=0)
    log_prob_masks = torch.stack(log_prob_masks, dim=0)
    # (batch_size, 3, 3), (batch_size, 3, 3), (batch_size, 3), (batch_size, 6)

    lattice_lengths = lattice_lengths.view(-1, 1, 3)
    lattice_angles = lattice_angles.view(-1, 1, 3)
    # (batch_size, 1, 3)

    constrained_lengths = torch.bmm(lattice_lengths, length_projection_matrices)
    constrained_angles = torch.bmm(lattice_angles, angle_projection_matrices)
    # (batch_size, 1, 3)

    constrained_lengths = constrained_lengths.squeeze(dim=1)
    constrained_angles = constrained_angles.squeeze(dim=1) + angle_translation_vectors
    # (batch_size, 3)

    lattice_masked_log_probs = log_prob_masks * lattice_log_probs

    return constrained_lengths, constrained_angles, lattice_masked_log_probs


def lattice_transform_and_log_prob_mask(
    spacegroup: int, device: Union[torch.device, str]
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Args:
        spacegroup: [1, 230]
        device:

    Returns:
        length_matrix: (3, 3)
        angle_matrix: (3, 3)
        angle_vector: (3,)
        log_prob_mask: (6,)

    Ex: for shape (1, 3) vector 'L' of lengths, and (1, 3) vector A of angles,
    and (6,) vector 'logPf' of log probabilities,
        constrained_lengths = L @ length_matrix
        constrained_angles = A @ angle_matrix + angle_vector
        masked_log_probs = log_prob_mask * logPf
    Note that `log_prob_mask` assumes the order of logPf corresponds to the
    following order of lattice parameters:
        (a, b, c, alpha, beta, gamma)
    """
    if 1.0 <= spacegroup <= 2.0:
        # Triclinic
        # (a, b, c, alpha, beta, gamma)
        length_matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
        )
        angle_matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
        )
        angle_vector = torch.tensor([0.0, 0.0, 0.0], device=device)
        log_prob_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
    elif 3 <= spacegroup <= 15:
        # Monoclinic
        # (a, b, c, 90., beta, 90.)
        length_matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
        )
        angle_matrix = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        angle_vector = torch.tensor([90.0, 0.0, 90.0], device=device)
        log_prob_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], device=device)
    elif 16 <= spacegroup <= 74:
        # Orthorhombic
        # (a, b, c, 90, 90, 90)
        length_matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
        )
        angle_matrix = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        angle_vector = torch.tensor([90.0, 90.0, 90.0], device=device)
        log_prob_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=device)
    elif 75 <= spacegroup <= 142:
        # Tetragonal
        # (a, a, c, 90, 90, 90)
        length_matrix = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device
        )
        angle_matrix = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        angle_vector = torch.tensor([90.0, 90.0, 90.0], device=device)
        log_prob_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device)
    # elif 143 <= spacegroup <= 167:
    #     # Trigonal/rhombohedral
    #     # (a, a, a, alpha, alpha, alpha)
    #     length_matrix = torch.tensor([[1., 1., 1.],
    #                                   [0., 0., 0.],
    #                                   [0., 0., 0.]], device=device)
    #     angle_matrix = torch.tensor([[1., 1., 1.],
    #                                  [0., 0., 0.],
    #                                  [0., 0., 0.]], device=device)
    #     angle_vector = torch.tensor([0., 0., 0.], device=device)
    #     log_prob_mask = torch.tensor([1., 0., 0., 1., 0., 0.], device=device)
    # elif 168 <= spacegroup <= 194:
    elif 143 <= spacegroup <= 194:
        # Trigonal (rhombohedral/hexagonal)
        # (a, a, c, 90, 90, 120)
        length_matrix = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1]], device=device
        )
        angle_matrix = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        angle_vector = torch.tensor([90.0, 90.0, 120.0], device=device)
        log_prob_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device)
    elif 195 <= spacegroup <= 230:
        # Cubic
        # (a, a, a, 90, 90, 90)
        length_matrix = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        angle_matrix = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        angle_vector = torch.tensor([90.0, 90.0, 90.0], device=device)
        log_prob_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    else:
        raise AttributeError
    return length_matrix, angle_matrix, angle_vector, log_prob_mask


def primitive_lattice_matrix_from_conventional_lattice_params(
    space_group_indices: Tensor,
    conventional_lattice_lengths: Tensor = None,
    conventional_lattice_angles: Tensor = None,
    conventional_lattice_matrix: Tensor = None,
    device: torch.device = "cpu",
) -> Tensor:
    """
    Args:
        conventional_lattice_lengths: (n_crystals, 3)
        conventional_lattice_angles:  (n_crystals, 3)
        space_group_indices: torch.long
            shape (n_crystals,)
        conventional_lattice_matrix: torch.float
            (n_crystals, 3, 3)
        device:


    Returns:
        shape (n_crystals, 3, 3) primitive lattice matrices
    """
    if conventional_lattice_matrix is None:
        conventional_lattice_matrix = lattice_params_to_matrix_torch(
            conventional_lattice_lengths, conventional_lattice_angles
        )  # (n_crystals, 3, 3)
    P_matrices = global_vars.conventional_to_primitive_P_matrices.to(device)[
        space_group_indices
    ]
    # (n_crystals, 3, 3)
    primitive_lattice_matrix = torch.bmm(P_matrices, conventional_lattice_matrix)
    return primitive_lattice_matrix


def construct_fully_connected_graphs_with_periodic_boundaries(
    cart_coords: Tensor,  # (n_atoms, 3)
    lattice_matrix: Tensor,  # (n_crystals, 3 lattice vecs, 3 coords)
    num_nodes_per_crystal: Tensor,  # (n_crystals,)
    device: Union[str, torch.device] = "cpu",
    break_minimum_edge_ties: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Modified from https://github.com/Open-Catalyst-Project/ocp/blob/cdd891e5a27c5ebdd3b9b9d20a401d558e502e08/ocpmodels/common/utils.py#L556

    Computes DIRECTED NxN graphs where edges are computed as minimum image
    distances under periodic boundary conditions. For each pair of nodes in the
    primal unit cell, use the smallest edge between them.
    Masks out self-loops and edges with query nodes as source nodes.
    Source nodes are translated to periodic cell images.

    Args:
        cart_coords: torch.float
            shape (n_atoms, 3)
        lattice_matrix: torch.float
            shape (n_crystals, 3, 3)
        num_nodes_per_crystal: torch.long
            shape (n_crystals,)
        break_minimum_edge_ties: torch.bool
        device:

    Returns:
        destination_index: torch.long
            Shape (n_edges,). Zero-indexed destination node indices.
        source_index: torch.long
            Shape (n_edges,). Zero-indexed source node indices.
        pbc_frac_offsets_per_source_atom: torch.float
            Shape (n_edges, 3). Fractional lattice translations of the source
            nodes.
        num_edges_per_crystal: torch.long
            Shape (batch_size,).
    """
    batch_size = len(num_nodes_per_crystal)

    # position of the atoms
    atom_pos = cart_coords  # (n_atoms, 3)

    # Before computing the pairwise distances between atoms, first create a list
    # of atom indices to compare for the entire batch
    # num_atoms_per_crystal = num_nodes_per_crystal
    num_atoms_per_crystal_sqr = (num_nodes_per_crystal**2).long()
    # (num_crystals,)

    # node index offset between crystals
    first_node_index_per_crystal = (
        torch.cumsum(num_nodes_per_crystal, dim=0) - num_nodes_per_crystal
    )
    first_node_index_per_crystal_expand = torch.repeat_interleave(
        first_node_index_per_crystal, num_atoms_per_crystal_sqr
    )
    num_atoms_per_crystal_expand = torch.repeat_interleave(
        num_nodes_per_crystal, num_atoms_per_crystal_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to
    # num_atoms_per_crystal_sqr for each crystal. The tensor will be used to
    # compute indices for pairs of atoms. This is a very convoluted way to
    # implement the following (but 10x faster since it removes the for loop):
    # atom_count_sqr = torch.tensor([])
    # for batch_idx in range(batch_size):
    #    atom_count_sqr = torch.cat([atom_count_sqr, torch.arange(num_atoms_per_crystal_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_crystal_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_crystal_sqr, dim=0) - num_atoms_per_crystal_sqr
    )
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_crystal_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute indices for pairs of atoms using division and mod. If systems get
    # too large this approach could run into numerical precision issues
    destination_index = (
        torch.div(atom_count_sqr, num_atoms_per_crystal_expand, rounding_mode="floor")
    ) + first_node_index_per_crystal_expand
    source_index = (
        atom_count_sqr % num_atoms_per_crystal_expand
    ) + first_node_index_per_crystal_expand
    # 3 atoms example:
    # destination_index: [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # source_index:      [0, 1, 2, 0, 1, 2, 0, 1, 2]
    map_edge_to_crystal = torch.arange(
        batch_size, device=device
    ).repeat_interleave(num_atoms_per_crystal_sqr, dim=0)
    # same shape as source_index and destination_index

    n_edges_per_crystal_before_masking = num_atoms_per_crystal_sqr
    # (n_crystals,)

    # Get the positions for each node
    source_position = torch.index_select(atom_pos, 0, source_index)
    destination_position = torch.index_select(atom_pos, 0, destination_index)

    # Only create 1 cell image in each lattice direction for efficiency
    num_supercell_images = len(OFFSET_LIST)
    supercell_frac_offsets = torch.tensor(
        OFFSET_LIST, dtype=torch.float, device=device
    )  # todo: pre-compute this tensor
    # (num_supercell_images, 3)
    batch_supercell_frac_offsets = supercell_frac_offsets.view(
        1, num_supercell_images, 3
    ).expand(batch_size, num_supercell_images, 3)
    # (batch_size, num_supercell_images, 3)
    pbc_frac_offsets_per_source_atom = batch_supercell_frac_offsets.repeat_interleave(
        n_edges_per_crystal_before_masking, dim=0
    )  # (num_atom_pairs, num_supercell_images, 3)
    pbc_cart_offsets_per_source_atom = torch.bmm(
        pbc_frac_offsets_per_source_atom,
        lattice_matrix.repeat_interleave(
            n_edges_per_crystal_before_masking, dim=0
        ),  # (num_atom_pairs, 3 lattice vecs, 3 coords)
    )  # (num_atom_pairs, num_supercell_images, 3)

    destination_position = destination_position.unsqueeze(1).expand(
        -1, num_supercell_images, -1
    )
    source_position = source_position.unsqueeze(1).expand(-1, num_supercell_images, -1)
    source_position = source_position + pbc_cart_offsets_per_source_atom
    # (num_atom_pairs, num_supercell_images, 3)

    source_index = source_index.view(-1, 1).repeat(1, num_supercell_images)
    destination_index = destination_index.view(-1, 1).repeat(1, num_supercell_images)
    map_edge_to_crystal = map_edge_to_crystal.view(-1, 1).expand(-1, num_supercell_images)
    # (num_atom_pairs, num_supercell_images)

    inter_atom_distances = (source_position - destination_position).norm(dim=-1)
    # (num_atom_pairs, num_supercell_images)

    # Remove self-loops
    mask = torch.logical_or(
        source_index != destination_index, torch.gt(inter_atom_distances, 1e-5)
    )

    # Remove pairs that are too far apart
    destination_index = torch.masked_select(destination_index, mask)
    source_index = torch.masked_select(source_index, mask)
    map_edge_to_crystal = torch.masked_select(map_edge_to_crystal, mask)
    pbc_frac_offsets_per_source_atom = torch.masked_select(
        pbc_frac_offsets_per_source_atom, mask.unsqueeze(-1).expand(-1, -1, 3)
    ).view(-1, 3)
    pbc_cart_offsets_per_source_atom = torch.masked_select(
        pbc_cart_offsets_per_source_atom, mask.unsqueeze(-1).expand(-1, -1, 3)
    ).view(-1, 3)

    source_position = torch.index_select(atom_pos, 0, source_index)
    destination_position = torch.index_select(atom_pos, 0, destination_index)
    inter_atom_distances = (
        destination_position - source_position - pbc_cart_offsets_per_source_atom
    ).norm(dim=-1)

    (
        destination_index,                  # (num_minimal_edges,)
        source_index,                       # (num_minimal_edges,)
        pbc_frac_offsets_per_source_atom,   # (num_minimal_edges, 3)
        num_edges_per_crystal,              # (num_crystals,)
    ) = get_smallest_edge_per_primal_node_pair(
        dst_idx=destination_index,
        src_idx=source_index,
        map_edge_to_crystal=map_edge_to_crystal,
        inter_atom_distances=inter_atom_distances,
        pbc_frac_offsets_per_source_atom=pbc_frac_offsets_per_source_atom,
        num_nodes_per_crystal=num_nodes_per_crystal,
        break_minimum_edge_ties=break_minimum_edge_ties,
    )
    return (
        destination_index,
        source_index,
        pbc_frac_offsets_per_source_atom,
        num_edges_per_crystal,
    )


def get_smallest_edge_per_primal_node_pair(
    dst_idx: Tensor,
    src_idx: Tensor,
    map_edge_to_crystal: Tensor,
    inter_atom_distances: Tensor,
    pbc_frac_offsets_per_source_atom: Tensor,
    num_nodes_per_crystal: Tensor,
    break_minimum_edge_ties: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    For each pair of nodes in the primal unit cell, get the smallest edge
    between them under periodic boundary conditions. If not
    'break_minimum_edge_ties', includes edges with the same length as the
    smallest edge.

    Args:
        dst_idx: torch.long
            shape (num_edges,)
        src_idx: torch.long
            shape (num_edges,)
        map_edge_to_crystal: torch.long
            shape (num_edges,)
        inter_atom_distances: torch.float
            shape (num_edges,)
        pbc_frac_offsets_per_source_atom: torch.float
            shape (num_edges, 3)
        num_nodes_per_crystal: torch.float
            shape (num_crystals,)

    Returns:
        dst_idx: shape (num_minimal_edges,)
        src_idx: shape (num_minimal_edges,)
        pbc_frac_offsets_per_source_atom: shape (num_minimal_edges,)
        num_edges_per_crystal: shape (num_crystals,)
    """
    edges = torch.stack((dst_idx, src_idx), dim=-1)
    # (num_edges, 2)
    unique_edges, map_edge_to_unique_edge = torch.unique(
        edges, dim=0, return_inverse=True
    )
    # (num_unique_edges, 2), (num_edges,)
    min_inter_atom_distances, smallest_edge_idxs = scatter_min(
        src=inter_atom_distances,
        index=map_edge_to_unique_edge,
        dim_size=len(unique_edges),
    )
    # (num_unique_edges,), (num_unique_edges,)
    if break_minimum_edge_ties:
        edges = edges[smallest_edge_idxs]  # (num_unique_edges,)
        pbc_frac_offsets_per_source_atom = (
            pbc_frac_offsets_per_source_atom[smallest_edge_idxs]
        )  # (num_unique_edges, 3)
        num_edges_per_crystal = (num_nodes_per_crystal ** 2).long()
        # (num_crystals,)
    else:
        # -- Keep edge if it's within 1e-4 Angstroms of the minimum edge length
        cutoff_distances = 1e-4 + min_inter_atom_distances[map_edge_to_unique_edge]
        # (num_edges,)
        keep_edge_mask = inter_atom_distances < cutoff_distances
        indices_of_edges_to_keep = keep_edge_mask.nonzero().view(-1)
        # (num_edges_to_keep,)
        edges = edges[indices_of_edges_to_keep]
        # (num_edges_to_keep, 2)
        pbc_frac_offsets_per_source_atom = (
            pbc_frac_offsets_per_source_atom[indices_of_edges_to_keep]
        )  # (num_edges_to_keep, 3)

        batch_size = num_nodes_per_crystal.shape[0]
        num_edges_per_crystal = scatter(
            src=torch.ones(edges.shape[0], device=edges.device),
            index=map_edge_to_crystal[indices_of_edges_to_keep],
            dim=-1,
            dim_size=batch_size,
            reduce="sum",
        ).long()  # (num_crystals,)

    dst_idx = edges[:, 0]  # (num_unique_edges,)
    src_idx = edges[:, 1]  # (num_unique_edges,)
    return (
        dst_idx,
        src_idx,
        pbc_frac_offsets_per_source_atom,
        num_edges_per_crystal,
    )


def ocp_get_pbc_distances(
    coords,
    source_id,
    destination_id,
    lattice,
    pbc_frac_offsets_per_source_node,
    num_edges_per_crystal,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
):
    """Modified from https://github.com/Open-Catalyst-Project/ocp/blob/cdd891e5a27c5ebdd3b9b9d20a401d558e502e08/ocpmodels/common/utils.py#L513"""
    # correct for pbc
    neighbors = num_edges_per_crystal.to(lattice.device)
    lattice = torch.repeat_interleave(lattice, neighbors, dim=0)
    offsets = (
        pbc_frac_offsets_per_source_node.float()
        .view(-1, 1, 3)
        .bmm(lattice.float())
        .view(-1, 3)
    )
    distance_vectors = coords[source_id] + offsets - coords[destination_id]

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # # redundancy: remove zero distances
    # nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    # edge_index = torch.stack([source_id, destination_id], dim=0)[:, nonzero_idx]
    # distances = distances[nonzero_idx]
    # distance_vectors = distance_vectors[nonzero_idx]
    # offsets = offsets[nonzero_idx]
    edge_index = torch.stack([source_id, destination_id], dim=0)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def batched_convert_asu_frac_coords_to_primitive_cartesian_coords(
    asu_frac_coords: Tensor,
    asu_element_indices: Tensor,
    asu_wyckoff_indices: Tensor,
    n_coords_per_asu: Tensor,
    conventional_lattice_matrix: Tensor,
    space_group_indices: Tensor,
    device: Union[str, torch.device] = "cpu",
    return_cartesian_coords: bool = True,
    return_node_is_original: bool = False,
    map_frac_coords_to_0_1_unit_cell: bool = True,
    get_primitive_cell: bool = True,
) -> Tuple:
    """
    Args:
        asu_frac_coords: torch.float
            shape (n_asu_atoms_in_batch, 3). Conventional unit cell fractional
            coordinates in the asymmetric unit.
        asu_element_indices: torch.long
            shape (n_asu_atoms_in_batch,). Zero-indexed element types.
        asu_wyckoff_indices: torch.long
            shape (n_asu_atoms_in_batch,). Zero-indexed Wyckoff positions.
        n_coords_per_asu: torch.long
            shape (B,). Number of atoms per asymmetric unit in the batch.
        conventional_lattice_matrix: torch.float
            shape (B, 3, 3)
        space_group_indices: torch.long
            shape (B,). Zero-indexed space groups in [0,229].
        device:
        get_primitive_cell: bool
            If True, return primitive cell information. Else, return
            conventional cell information.

    Returns:
        primitive_cartesian_coords: Tensor (torch.float)
            Shape (n_primitive_atoms_in_batch, 3).
        primitive_element_indices: Tensor (torch.long)
            Shape (n_primitive_atoms_in_batch,).
        primitive_wyckoff_indices: Tensor (torch.long)
            Shape (n_primitive_atoms_in_batch,).
        num_prim_nodes_per_crystal: Tensor (torch.long)
            Shape (n_crystals,).
        primitive_lattice_matrix: Tensor (torch.float)
            Shape (n_crystals, 3, 3).
        asu_frac_coords_of_prim_atoms: Tensor (torch.float)
            Shape (n_primitive_atoms_in_batch, 3). Conventional cell fractional
            coordinates in the asymmetric unit that each primitive atom
            originated from. These fractional coordinates are not necessarily
            in [0, 1] so that the ASU is contiguous.
    If return_node_is_original:
        node_is_original: Tensor (torch.bool)
            Shape (n_primitive_atoms_in_batch,)
    """
    batch_size: int = n_coords_per_asu.shape[0]
    num_asu_nodes_per_crystal = n_coords_per_asu

    # -- Init conventional to primitive cell conversions
    conventional_to_primitive_transformations: Tensor = (
        global_vars.conventional_to_primitive_invP_matrices.to(device)[space_group_indices]
    )
    # (B, 3, 3)

    # -- Get general Wyckoff position operations
    padded_general_wyckoff_matrices = global_vars.padded_general_wyckoff_matrices.to(device)[
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
        num_asu_nodes_per_crystal,
        dim=0,
        output_size=asu_frac_coords.shape[0],
    )
    # (n_asu_atoms_in_batch, 192, 3, 3)
    padded_general_wyckoff_translations = torch.repeat_interleave(
        padded_general_wyckoff_translations,
        num_asu_nodes_per_crystal,
        dim=0,
        output_size=asu_frac_coords.shape[0],
    )
    # (n_asu_atoms_in_batch, 192, 1, 3)
    padded_general_wyckoff_ops_mask = torch.repeat_interleave(
        padded_general_wyckoff_ops_mask,
        num_asu_nodes_per_crystal,
        dim=0,
        output_size=asu_frac_coords.shape[0],
    )
    # (n_asu_atoms_in_batch, 192)

    stacked_general_wyckoff_matrices = padded_general_wyckoff_matrices.masked_select(
        padded_general_wyckoff_ops_mask[:, :, None, None].expand(-1, -1, 3, 3)
    ).view(-1, 3, 3)
    # (n_orbited_atoms, 3, 3)
    stacked_general_wyckoff_translations = padded_general_wyckoff_translations.masked_select(
        padded_general_wyckoff_ops_mask[:, :, None, None].expand(-1, -1, 1, 3)
    ).view(-1, 1, 3)
    # (n_orbited_atoms, 1, 3)

    # --
    # Orbit all ASU atoms in the batch with general Wyckoff positions. Mod to
    # [0, 1] and then convert to primitive cells (unfortunately the mod operator
    # does not commute with matrix multiply so each operation must be separate).
    # --
    # - Apply general Wyckoff position operations to orbit asymmetric unit atoms
    #   into the conventional unit cell
    asu_frac_coords_repeated = asu_frac_coords.repeat_interleave(
        general_wyckoff_multiplicity_per_crystal.repeat_interleave(
            num_asu_nodes_per_crystal, dim=0, output_size=asu_frac_coords.shape[0]
        ),
        dim=0,
    )[:, None, :]
    # (n_orbited_atoms, 1, 3)
    conventional_frac_coords = (
        torch.bmm(asu_frac_coords_repeated, stacked_general_wyckoff_matrices)
        + stacked_general_wyckoff_translations
    )
    # (n_orbited_atoms, 1, 3)
    if map_frac_coords_to_0_1_unit_cell:
        conventional_frac_coords = conventional_frac_coords % 1.0

    # - Convert conventional to primitive fractional coordinates
    if get_primitive_cell:
        stacked_conventional_to_primitive_transformations = torch.repeat_interleave(
            conventional_to_primitive_transformations,
            num_asu_nodes_per_crystal * general_wyckoff_multiplicity_per_crystal,
            dim=0,
            output_size=conventional_frac_coords.shape[0],
        )
        # (n_orbited_atoms, 3, 3)
        primitive_frac_coords = (
            torch.bmm(
                conventional_frac_coords, stacked_conventional_to_primitive_transformations
            ).squeeze(dim=1)
        )
        # (n_orbited_atoms, 3)
    else:
        primitive_frac_coords = conventional_frac_coords.view(-1, 3)
    if map_frac_coords_to_0_1_unit_cell:
        primitive_frac_coords = primitive_frac_coords % 1.0
    # -

    map_node_to_crystal = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        num_asu_nodes_per_crystal * general_wyckoff_multiplicity_per_crystal,
        dim=0,
        output_size=primitive_frac_coords.shape[0],
    )
    # (n_orbited_atoms,)
    # -

    # --
    # De-duplicate identical atoms. Atoms are identical iff they originated from
    # the same ASU atom and overlap.
    # --
    # - We will effectively compute concatenated, flattened adjacency matrices,
    #   where each adjacency matrix is only between atoms of the same orbit.
    orbit_size_per_asu_atom = general_wyckoff_multiplicity_per_crystal.repeat_interleave(
        num_asu_nodes_per_crystal, dim=0, output_size=asu_frac_coords.shape[0]
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

    if return_node_is_original:
        # Create 'node_is_original' tensor BEFORE atom de-duplication
        node_is_original = torch.zeros(
            primitive_frac_coords.shape[0], dtype=torch.bool, device=device
        )
        # (n_orbited_atoms,)
        node_is_original[first_asu_atom_index_per_orbit] = True

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
    adjacency_row_atom_coords = primitive_frac_coords[atom_adjacency_row_index]
    adjacency_col_atom_coords = primitive_frac_coords[atom_adjacency_col_index]
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
    unique_non_overlapping_atom_indices = torch.unique(
        segment_coo(
            src=atom_adjacency_col_index[overlapping_atom_pairs_mask],
            index=atom_adjacency_row_index[overlapping_atom_pairs_mask],
            reduce="min",
        )
    )
    # (n_unique_orbited_atoms,)

    # De-duplicate atom coordinates. 'unique_non_overlapping_atom_indices'
    # indexes into shape (n_orbited_atoms, *) tensors
    primitive_frac_coords = primitive_frac_coords[unique_non_overlapping_atom_indices]
    # (n_unique_orbited_atoms, 3)
    map_node_to_crystal = map_node_to_crystal[unique_non_overlapping_atom_indices]
    # (n_unique_orbited_atoms,)
    if return_node_is_original:
        node_is_original = node_is_original[unique_non_overlapping_atom_indices]
        # (n_unique_orbited_atoms,)
    num_prim_nodes_per_crystal = segment_coo(
        src=torch.ones_like(map_node_to_crystal), index=map_node_to_crystal, reduce="sum"
    )
    # (B,)
    assert num_prim_nodes_per_crystal.shape[0] == batch_size

    # Get primitive cell Wyckoff and element indices from the ASU
    map_prim_coords_to_asu_coord = torch.arange(
        asu_frac_coords.shape[0], device=device
    ).repeat_interleave(
        general_wyckoff_multiplicity_per_crystal.repeat_interleave(
            num_asu_nodes_per_crystal, dim=0, output_size=asu_frac_coords.shape[0]
        ),
        dim=0,
        output_size=conventional_frac_coords.shape[0],
    )[unique_non_overlapping_atom_indices]
    # (n_unique_orbited_atoms,)
    primitive_wyckoff_indices = asu_wyckoff_indices[map_prim_coords_to_asu_coord]
    # (n_unique_orbited_atoms,)
    primitive_element_indices = asu_element_indices[map_prim_coords_to_asu_coord]
    # (n_unique_orbited_atoms,)
    asu_frac_coords_of_prim_atoms = asu_frac_coords[map_prim_coords_to_asu_coord]
    # (n_unique_orbited_atoms,)

    # -- Convert fractional coordinates to Cartesian
    primitive_lattice_matrix = primitive_lattice_matrix_from_conventional_lattice_params(
        space_group_indices=space_group_indices,
        conventional_lattice_matrix=conventional_lattice_matrix,
        device=device,
    )  # (n_crystals, 3, 3)

    if return_cartesian_coords:
        primitive_coords = frac_to_cart_coords(
            primitive_frac_coords,
            num_prim_nodes_per_crystal,
            lattice_matrix=primitive_lattice_matrix,
        )
    else:
        primitive_coords = primitive_frac_coords

    if return_node_is_original:
        return (
            primitive_coords,  # (n_primitive_atoms_in_batch, 3)
            primitive_element_indices,  # (n_primitive_atoms_in_batch,)
            primitive_wyckoff_indices,  # (n_primitive_atoms_in_batch,)
            num_prim_nodes_per_crystal,  # (n_crystals,)
            primitive_lattice_matrix,  # (n_crystals, 3, 3)
            map_prim_coords_to_asu_coord,  # (n_primitive_atoms_in_batch,)
            node_is_original,  # (n_primitive_atoms_in_batch,)
            asu_frac_coords_of_prim_atoms,  # (n_primitive_atoms_in_batch, 3)
        )
    else:
        return (
            primitive_coords,  # (n_primitive_atoms_in_batch, 3)
            primitive_element_indices,  # (n_primitive_atoms_in_batch,)
            primitive_wyckoff_indices,  # (n_primitive_atoms_in_batch,)
            num_prim_nodes_per_crystal,  # (n_crystals,)
            primitive_lattice_matrix,  # (n_crystals, 3, 3)
            map_prim_coords_to_asu_coord,  # (n_primitive_atoms_in_batch,)
            asu_frac_coords_of_prim_atoms,  # (n_primitive_atoms_in_batch, 3)
        )


def get_orbits_from_asu_positions(
    asu_frac_coords: tensor, space_group_number: int
) -> List[tensor]:
    """
    Args:
        asu_frac_coords: shape (n_points_in_asu, 3)
        space_group_number: 1-indexed space group number in [1, 230]

    Returns:
        Length-(n_points_in_asu) list of tensors, where entry i is a tensor of
        shape (atom i's Wyckoff multiplicity, 3)
    """
    assert 1 <= space_group_number <= 230
    device = asu_frac_coords.device
    asu_frac_coords = asu_frac_coords % 1.0

    pyxtal_space_group = global_vars.pyxtal_space_group_dict[space_group_number]
    general_wyckoff_position = pyxtal_space_group.Wyckoff_positions[0]
    rotations: Tensor = general_wyckoff_position.ops["rotations"].to(device)
    # (general_wyckoff_position.multiplicity, 3, 3)
    translations: Tensor = general_wyckoff_position.ops["translations"].to(device)
    # (general_wyckoff_position.multiplicity, 1, 3)

    # -- Orbit all ASU points with the general Wyckoff position simultaneously
    orbits = (asu_frac_coords[None, :, :] @ rotations + translations).transpose(0, 1) % 1.0
    # (n_points_in_asu, general_wyckoff_position.multiplicity, 3)

    # -- De-duplicate overlapping atoms with the same asymmetric unit position.
    # Yay batched outer subtraction. Use mod to handle periodic boundaries.
    position_diffs = torch.abs(
        (orbits[..., None, :] - orbits[..., None, :, :] + 0.5) % 1.0 - 0.5
    )
    # (n_points_in_asu, general_wyckoff_position.multiplicity, general_wyckoff_position.multiplicity, 3)

    duplicates_adjacency = torch.all(position_diffs < 1e-4, dim=-1)
    # (n_points_in_asu, general_wyckoff_position.multiplicity, general_wyckoff_position.multiplicity)

    # For each atom, get the smallest index of the atom in its orbit which
    # overlaps with it. Use torch.argmax, which breaks ties by returning the
    # first appearance. Note that we will still have duplicate IDs after this
    # operation and require a torch.unique() call later.
    unique_atom_idxs = torch.argmax(duplicates_adjacency.float(), dim=-1)
    # (n_points_in_asu, general_wyckoff_position.multiplicity)

    # Each ASU atom may have a different orbit size depending on its Wyckoff
    # position, so I think a loop is unavoidable here.
    deduped_orbits: List[tensor] = [
        orbits[i][torch.unique(unique_atom_idxs[i])] for i in range(asu_frac_coords.shape[0])
    ]
    return deduped_orbits


# TODO: deprecate this function in favor of get_orbits_from_asu_positions()
def get_orbit_from_single_asu_position(
    asu_frac_coord: tensor, frac_coord_wyckoff_index: int, space_group_number: int
) -> tensor:
    """
    Yields full orbit of a single fractional coordinate in the ASU. Puts the
    orbit into the default Hall settings used by the International Tables of
    Crystallography (adopted by cctbx and pyxtal).

    Args:
        asu_frac_coord: shape (3,) tensor
        frac_coord_wyckoff_index: Zero-indexed Wyckoff position that
            'asu_frac_coord' resides in. Assumes ascending Wyckoff multiplicity
            (letters ordered alphabetically).
        space_group_number: 1-indexed

    Returns:
        shape (frac_coord_wyckoff_position.multiplicity, 3)
    """
    pyxtal_space_group = global_vars.pyxtal_space_group_dict[space_group_number]

    # pyxtal orders its Wyckoff positions with descending multiplicity (opposite
    # of alphabetically), whereas ours are in ascending order
    frac_coord_wyckoff_position = pyxtal_space_group.Wyckoff_positions[
        len(pyxtal_space_group.Wyckoff_positions) - frac_coord_wyckoff_index - 1
    ]
    general_wyckoff_position = pyxtal_space_group.Wyckoff_positions[0]

    device = asu_frac_coord.device
    asu_frac_coord = asu_frac_coord % 1.0
    # asu_frac_coord = convert_cctbx_asu_frac_coords_to_spglib_hall_number(space_group_number, asu_frac_coord)

    rotations: Tensor = general_wyckoff_position.ops["rotations"].to(device)
    # (general_wyckoff_position.multiplicity, 3, 3)
    translations: Tensor = general_wyckoff_position.ops["translations"].to(device)
    # (general_wyckoff_position.multiplicity, 1, 3)
    orbit = (asu_frac_coord[None, None, :] @ rotations + translations).view(-1, 3) % 1.0
    # (general_wyckoff_position.multiplicity, 3)

    if frac_coord_wyckoff_position.letter != general_wyckoff_position.letter:
        # Remove duplicates
        special_wyckoff_position = frac_coord_wyckoff_position

        position_diffs = torch.abs(
            ((orbit - orbit[:, None] + 0.5) % 1.0) - 0.5
        )  # (general_wyckoff_position.multiplicity, general_wyckoff_position.multiplicity, 3)

        duplicates_adjacency = torch.all(position_diffs < 1e-4, dim=2)

        # For each atom, get the smallest index of the atom which overlaps with
        # that atom. Keep the resulting unique indices.
        unique_atom_idxs = torch.unique(torch.argmax(duplicates_adjacency.float(), dim=1))
        # torch.argmax breaks ties by returning the first appearance

        orbit = orbit[unique_atom_idxs]
        if orbit.shape[0] != special_wyckoff_position.multiplicity:
            warnings.warn(
                f"Found orbit shape {orbit.shape[0]} for space group "
                f"{special_wyckoff_position.number}, wyckoff position "
                f"{special_wyckoff_position.multiplicity}{frac_coord_wyckoff_position.letter}. "
                f"The orbited fractional coordinate was {asu_frac_coord}."
            )
    return orbit  # (frac_coord_wyckoff_position.multiplicity, 3)


def convert_cctbx_asu_frac_coords_to_spglib_hall_number(
    space_group_number: int, cctbx_asu_frac_coords: tensor, inverse_transform: bool = False
) -> tensor:
    """
    Convert fractional coordinates in ITA
    Args:
        space_group_number:
        cctbx_asu_frac_coords:
        inverse_transform: Instead of converting cctbx setting to spglib
            setting, do the inverse.

    Returns:

    """
    device = cctbx_asu_frac_coords.device
    space_groups_with_different_hall_numbers = [
        48,
        50,
        59,
        68,
        70,
        85,
        86,
        88,
        125,
        126,
        129,
        130,
        133,
        134,
        137,
        138,
        141,
        142,
        201,
        203,
        222,
        224,
        227,
        228,
    ]
    if space_group_number not in space_groups_with_different_hall_numbers:
        return cctbx_asu_frac_coords
    elif space_group_number == 48:
        # pyxtal/cctbx uses Hall 229, spglib uses 228
        translation = torch.tensor([-1 / 4.0, -1 / 4.0, -1 / 4.0], device=device)
    elif space_group_number == 50:
        # pyxtal/cctbx uses Hall 234, spglib uses 233
        translation = torch.tensor([-1 / 4.0, -1 / 4.0, 0.0], device=device)
    elif space_group_number == 59:
        # pyxtal/cctbx uses Hall 279, spglib uses 278
        translation = torch.tensor([-1 / 4.0, -1 / 4.0, 0.0], device=device)
    elif space_group_number == 68:
        # pyxtal/cctbx uses Hall 323, spglib uses 322
        translation = torch.tensor([0.0, -1 / 4.0, -1 / 4.0], device=device)
    elif space_group_number == 70:
        # pyxtal/cctbx uses Hall 336, spglib uses 335
        translation = torch.tensor([1 / 8.0, 1 / 8.0, 1 / 8.0], device=device)
    elif space_group_number == 85:
        # pyxtal/cctbx uses Hall 360, spglib uses 359
        translation = torch.tensor([1 / 4.0, -1 / 4.0, 0.0], device=device)
    elif space_group_number == 86:
        # pyxtal/cctbx uses Hall 362, spglib uses 361
        translation = torch.tensor([1 / 4.0, 1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 88:
        # pyxtal/cctbx uses Hall 365, spglib uses 364
        translation = torch.tensor([0.0, 1 / 4.0, 1 / 8.0], device=device)
    elif space_group_number == 125:
        # pyxtal/cctbx uses Hall 403, spglib uses 402
        translation = torch.tensor([1 / 4.0, 1 / 4.0, 0.0], device=device)
    elif space_group_number == 126:
        # pyxtal/cctbx uses Hall 405, spglib uses 404
        translation = torch.tensor([1 / 4.0, 1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 129:
        # pyxtal/cctbx uses Hall 409, spglib uses 408
        translation = torch.tensor([1 / 4.0, -1 / 4.0, 0.0], device=device)
    elif space_group_number == 130:
        # pyxtal/cctbx uses Hall 411, spglib uses 410
        translation = torch.tensor([1 / 4.0, -1 / 4.0, 0.0], device=device)
    elif space_group_number == 133:
        # pyxtal/cctbx uses Hall 415, spglib uses 414
        translation = torch.tensor([1 / 4.0, -1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 134:
        # pyxtal/cctbx uses Hall 417, spglib uses 416
        translation = torch.tensor([1 / 4.0, -1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 137:
        # pyxtal/cctbx uses Hall 421, spglib uses 420
        translation = torch.tensor([1 / 4.0, -1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 138:
        # pyxtal/cctbx uses Hall 423, spglib uses 422
        translation = torch.tensor([1 / 4.0, -1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 141:
        # pyxtal/cctbx uses Hall 427, spglib uses 426
        translation = torch.tensor([0.0, -1 / 4.0, 1 / 8.0], device=device)
    elif space_group_number == 142:
        # pyxtal/cctbx uses Hall 429, spglib uses 428
        translation = torch.tensor([0.0, -1 / 4.0, 1 / 8.0], device=device)
    elif space_group_number == 201:
        # pyxtal/cctbx uses Hall 496, spglib uses 495
        translation = torch.tensor([1 / 4.0, 1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 203:
        # pyxtal/cctbx uses Hall 499, spglib uses 498
        translation = torch.tensor([1 / 8.0, 1 / 8.0, 1 / 8.0], device=device)
    elif space_group_number == 222:
        # pyxtal/cctbx uses Hall 519, spglib uses 518
        translation = torch.tensor([1 / 4.0, 1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 224:
        # pyxtal/cctbx uses Hall 522, spglib uses 521
        translation = torch.tensor([1 / 4.0, 1 / 4.0, 1 / 4.0], device=device)
    elif space_group_number == 227:
        # pyxtal/cctbx uses Hall 526, spglib uses 525
        translation = torch.tensor([1 / 8.0, 1 / 8.0, 1 / 8.0], device=device)
    elif space_group_number == 228:
        # pyxtal/cctbx uses Hall 528, spglib uses 527
        translation = torch.tensor([3 / 8.0, 3 / 8.0, 3 / 8.0], device=device)

    if inverse_transform:
        return cctbx_asu_frac_coords - translation
    else:
        return cctbx_asu_frac_coords + translation


@torch.no_grad()
def asu_to_pymatgen_structure(asu: ASUCrystal) -> pmg_structure:
    # TODO: parallelize this function?
    conventional_lattice_matrix = lattice_params_to_matrix_torch(
        asu.conventional_lattice_lengths.double().unsqueeze(0),
        asu.conventional_lattice_angles.double().unsqueeze(0),
    )[0]  # (3, 3)
    pmg_conventional_lattice = Lattice(matrix=conventional_lattice_matrix.cpu().numpy())

    conventional_frac_coords, conventional_atomic_numbers = [], []
    for asu_frac_coord, wyckoff_index, atom_type in zip(
        asu.conventional_frac_coords, asu.wyckoff_indices, asu.element_indices
    ):
        atom_orbit: tensor = get_orbit_from_single_asu_position(
            asu_frac_coord,
            frac_coord_wyckoff_index=wyckoff_index.item(),
            space_group_number=int(asu.space_group_number),
        )  # (wyckoff_multiplicity, 3)
        conventional_atomic_numbers.append((atom_type + 1).repeat(atom_orbit.shape[0]))
        conventional_frac_coords.append(atom_orbit)

    conventional_atomic_numbers = (
        torch.cat(conventional_atomic_numbers, dim=0).flatten().cpu().tolist()
    )
    conventional_frac_coords = torch.cat(conventional_frac_coords, dim=0).cpu().numpy()
    structure = Structure(
        lattice=pmg_conventional_lattice,
        species=conventional_atomic_numbers,
        coords=conventional_frac_coords,
        to_unit_cell=True,
        coords_are_cartesian=False,
    )
    return structure


@torch.no_grad()
def pyxtal_pymatgen_structure_to_asu(
    structure: pmg_structure,
    symprec: float = 0.1,  # 0.1 - pymatgen default
    angle_tolerance: float = 5.0,  # 5.0 - pymatgen default
    device: Union[str, torch.device] = "cpu",
    assume_p1_only: bool = False,
    structure_to_fall_back_to: pmg_structure = None,
) -> ASUCrystal:
    # # Don't use pyxtal_cell.from_seed() because its call to pyxtal's
    #  Wyckoff_position.search_generator() further symmetrizes an
    #  already-symmetrized spglib object
    # pyxtal_cell = pyxtal.pyxtal()
    # pyxtal_cell.from_seed(structure, style='pyxtal', a_tol=angle_tolerance, tol=symprec)
    # space_group_number = pyxtal_cell.group.number
    # pyxtal_conventional_lattice = Lattice(matrix=pyxtal_cell.lattice.matrix)
    # conventional_lattice_lengths = torch.tensor(pyxtal_conventional_lattice.lengths, dtype=torch.float, device=device)
    # conventional_lattice_angles = torch.tensor(pyxtal_conventional_lattice.angles, dtype=torch.float, device=device)

    if assume_p1_only:
        space_group_number = 1
        structure = (
            structure.get_reduced_structure()
            if structure_to_fall_back_to is None else structure_to_fall_back_to
        )
        atom_orbits = [
            (
                frac_coord.reshape(1, 3) % 1.0,  # (1, 3)
                "a",  # Wyckoff letter
                atomic_number,  # atomic number
            )
            for frac_coord, atomic_number in zip(
                structure.frac_coords, structure.atomic_numbers
            )
        ]
        conventional_lattice_lengths = torch.tensor(
            structure.lattice.parameters[:3], dtype=torch.float, device=device
        )
        conventional_lattice_angles = torch.tensor(
            structure.lattice.parameters[3:], dtype=torch.float, device=device
        )
    else:
        (
            conventional_symmetrized_structure,
            space_group_number,
        ) = pyxtal.util.get_symmetrized_pmg(
            structure, tol=symprec, a_tol=angle_tolerance, style="pyxtal"
        )
        atom_orbits = [
            (
                np.stack(
                    [site.frac_coords for site in orbit], axis=0
                ),  # array: (wyckoff_multiplicity, 3)
                wyckoff_symbol[-1],  # str: wyckoff_letter, e.g. '48h'[-1] -> 'h'
                orbit[0].specie.Z,  # int: atomic number
            )
            for orbit, wyckoff_symbol in zip(
                conventional_symmetrized_structure.equivalent_sites,
                conventional_symmetrized_structure.wyckoff_symbols,
            )
        ]
        conventional_lattice_lengths = torch.tensor(
            conventional_symmetrized_structure.lattice.lengths,
            dtype=torch.float,
            device=device,
        )
        conventional_lattice_angles = torch.tensor(
            conventional_symmetrized_structure.lattice.angles, dtype=torch.float, device=device
        )

    # Check that lattice parameters are consistent with the space group
    _lengths, _angles, _ = torch_legal_lattice_parameters(
        space_groups=torch.tensor([space_group_number], dtype=torch.long, device=device),
        lattice_parameters=torch.cat(
            [conventional_lattice_lengths, conventional_lattice_angles], dim=0
        ).unsqueeze(0),
        lattice_log_probs=torch.zeros((1, 6)),
        device="cpu",
    )
    assert torch.equal(_lengths.squeeze(), conventional_lattice_lengths) and torch.equal(
        _angles.squeeze(), conventional_lattice_angles
    ), (
        f"Pymatgen structure (SG {space_group_number}) had conventional "
        f"lattice parameters, "
        f"{conventional_lattice_lengths, conventional_lattice_angles}, "
        f"which do not match the Bravais lattice constraints."
    )

    # atomic_numbers, frac_coords, asymmetric_mapping, symmops, multiplicities = [], [], [], [], []
    asu_atomic_numbers, asu_frac_coords, asu_wyckoff_letters = [], [], []
    asu_atom_wyckoff_shape_indices = []
    for orbit_frac_coords, orbit_wyckoff_letter, orbit_atomic_number in atom_orbits:
        # -- Get images of orbit in 3x3x3 supercell. Keep atom inside the ASU
        # 3x3x3 supercell = center cell + 26 images
        cell_translations = (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [-1, 0, 1],
                    [1, 0, -1],
                    [-1, 0, -1],
                    [0, -1, 1],
                    [0, 1, -1],
                    [0, -1, -1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [1, -1, -1],
                    [-1, -1, -1],
                ]
            )
            + Fraction()
        )
        orbit_rational_frac_coords = (orbit_frac_coords % 1.0) + Fraction()
        orbit_supercell_frac_coords = (
            orbit_rational_frac_coords + cell_translations[:, None, :]
        ).reshape(-1, 3)
        try:
            (
                frac_position_in_asu, intersecting_wyckoff_shape_index
            ) = intersect_coords_with_asu_wyckoff(
                orbit_supercell_frac_coords, orbit_wyckoff_letter, space_group_number
            )
        except AttributeError:
            from utils.io_utils import get_random_hash, get_project_dir
            # failed_structure_filename = get_project_dir() + "/structure" + get_random_hash() + ".cif"
            # structure.to(filename=failed_structure_filename, fmt="cif")
            # warnings.warn(
            #     f"Spglib symmetry finder failed. Saving CIF to {failed_structure_filename}. "
            #     "Identifying crystal with P1 space group."
            # )
            warnings.warn("Spglib symmetry finder failed. Identifying crystal with P1 space group.")
            if assume_p1_only:
                raise AttributeError
            else:
                return pyxtal_pymatgen_structure_to_asu(
                    (
                        structure.get_primitive_structure()
                        if structure_to_fall_back_to is None
                        else structure_to_fall_back_to
                    ),
                    symprec,
                    angle_tolerance,
                    device,
                    assume_p1_only=True,
                )

        asu_atomic_numbers.append(orbit_atomic_number)
        asu_frac_coords.append(frac_position_in_asu)
        asu_wyckoff_letters.append(orbit_wyckoff_letter)
        asu_atom_wyckoff_shape_indices.append(intersecting_wyckoff_shape_index)

    asu_frac_coords = torch.stack(asu_frac_coords, dim=0)
    asu_wyckoff_indices = torch.tensor(
        [
            ord(letter) - 97 if ord(letter) >= 97 else ord(letter) - 39
            for letter in asu_wyckoff_letters
        ],
        dtype=torch.long,
        device=device,
    )
    asu_atom_types = torch.tensor(asu_atomic_numbers, dtype=torch.long, device=device) - 1
    composition_space = torch.zeros(NUM_ELEMENTS, device=device)
    composition_space[asu_atom_types] = 1.0
    asu_atom_wyckoff_shape_indices = torch.tensor(asu_atom_wyckoff_shape_indices, dtype=torch.long, device=device)
    asu = ASUCrystal(
        space_group_number=torch.tensor(space_group_number, dtype=torch.long, device=device),
        composition_space=composition_space,
        conventional_lattice_lengths=conventional_lattice_lengths,
        conventional_lattice_angles=conventional_lattice_angles,
        element_indices=asu_atom_types,
        wyckoff_indices=asu_wyckoff_indices,
        conventional_frac_coords=asu_frac_coords,
        wyckoff_shape_indices=asu_atom_wyckoff_shape_indices,
    )
    return asu


def intersect_coords_with_asu_wyckoff(
    atom_orbit_frac_coords: np.array,
    wyckoff_letter: str,
    space_group_number: int,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[tensor, int]:
    """
    Args:
        atom_orbit_frac_coords: shape (n_positions, 3)
        wyckoff_letter: Letter indicating the Wyckoff position that
            'atom_orbit_frac_coords' resides in
        space_group_number: Integer in [1, 230]

    Returns:
        (shape (3,) array, integer)
    """
    ## -- This commented-out method does not always work because asu.is_inside()
    ##      relies on evaluating exact inequalities, but we only have floats
    # frac_position_in_asu = None
    # for frac_position in orbit_supercell_frac_coords:
    #     if asu.is_inside(frac_position):
    #         assert frac_position_in_asu is None, \
    #             'Only 1 atom from the orbit can be in the ASU, but multiple ' \
    #             'were found.'
    #         frac_position_in_asu = frac_position
    #         print(f'Dim-{atom_orbit.wp.get_dof()} Wyckoff')
    #         # break
    ## --
    def _is_inside(x: ndarray, hull_equations: ndarray, epsilon: float = 0.0) -> bool:
        """
        Args:
          x: shape (3,)
          hull_equations: shape (n_linear_shape_bounds, 4)
        """
        # (n_linear_shape_bounds, 3) @ (3,) = (n_linear_shape_bounds,)
        return np.all(
            hull_equations[:, :-1] @ x < -hull_equations[:, -1] + np.abs(epsilon)
        )

    asu_wyckoff_dict = global_vars.asu_wyckoff_dict[str(space_group_number)][wyckoff_letter]
    wyckoff_dof = int(asu_wyckoff_dict["dim"])
    if wyckoff_dof == 0:
        # Return the point in 'frac_positions' closest to the Wyckoff point.
        # Ties are broken by choosing the first closest.
        atom_orbit_frac_coords_tensor = torch.tensor(
            atom_orbit_frac_coords.astype("float32"), device=device
        )
        wyckoff_point = torch.tensor(
            asu_wyckoff_dict["vertices"].astype("float32"), device=device
        ).squeeze()
        # (3,)
        dists_to_wyckoff = torch.sum(
            (atom_orbit_frac_coords_tensor - wyckoff_point) ** 2, dim=1
        ).sqrt()
        # (n_positions,)
        index_of_frac_position_in_asu = torch.argmin(dists_to_wyckoff)
        frac_coord_in_asu = atom_orbit_frac_coords_tensor[index_of_frac_position_in_asu]
        distance_to_asu_wyckoff = torch.min(dists_to_wyckoff)
        intersecting_wyckoff_shape_index = 0
    elif wyckoff_dof == 1:
        (
            frac_coord_in_asu,
            distance_to_asu_wyckoff,
            intersecting_wyckoff_shape_index,
        ) = _get_atom_on_wyckoff_line_segments(
            atom_orbit_frac_coords, asu_wyckoff_dict["vertices"], device
        )
    elif wyckoff_dof == 2:
        """
        Project atoms onto plane of Wyckoff facet. Do inside/outside test.
        Keep the one inside. If none found inside any Wyckoff facets, project
        atoms onto the Wyckoff facet edges.
        """
        from scipy.spatial import Delaunay, ConvexHull

        atom_orbit_frac_coords = atom_orbit_frac_coords.astype("float32")
        # (n_positions, 3)
        wyckoff_polygons = [face.astype("float32") for face in asu_wyckoff_dict["vertices"]]
        # len-n_faces List of (n_face_vertices, 3) arrays

        # --  Then do inside/outside test
        indices_of_projected_points_in_wyckoff = []
        all_projection_distances = []
        points_with_orthogonal_projection_on_wyckoff = []
        wyckoff_polygon_indices_of_atoms_with_projections_inside = []
        for i, polygon in enumerate(wyckoff_polygons):  # polygon shape: (n_polygon_vertices, 3)
            # Orthogonally project 'atom_orbit_frac_coords' point P to be
            # coplanar with a Wyckoff facet.
            point_on_plane = polygon[0, :][None, :]  # (1, 3)
            map_3d_to_2d, map_2d_to_3d, plane_normal = projection_map_3d_plane_to_2d(
                polygon[:3, :]
            )
            projected_coords, projection_distances = project_point_to_plane(
                atom_orbit_frac_coords, point_on_plane, plane_normal
            )
            # (n_positions, 3), (n_positions,)

            # Map 3d points coplanar with Wyckoff facet to 2D xy-plane
            polygon_2d = map_3d_to_2d(polygon)
            projected_coords_2d = map_3d_to_2d(projected_coords)
            assert np.allclose(polygon_2d[:, 2], 0.0, atol=1.0e-6) and np.allclose(
                projected_coords_2d[:, 2], 0.0, atol=1.0e-6
            ), f"{np.max(np.abs(polygon_2d[:, 2])), np.max(np.abs(projected_coords_2d[:, 2]))}"
            polygon_2d, projected_coords_2d = polygon_2d[:, :2], projected_coords_2d[:, :2]
            # (n_face_vertices, 2), (n_positions, 2)

            # Do inside/outside test
            hull = Delaunay(polygon_2d)
            result = hull.find_simplex(
                projected_coords_2d, tol=1e-6
            )  # tol=100 * np.finfo(np.double).eps
            # (n_positions,) array of np.int32
            indices_of_projected_points_in_polygon = (result >= 0).nonzero()[
                0
            ]  # (n_projected_points_in_polygon,)
            indices_of_projected_points_in_wyckoff.append(
                indices_of_projected_points_in_polygon
            )

            all_projection_distances.append(
                projection_distances[indices_of_projected_points_in_polygon]
            )
            points_with_orthogonal_projection_on_wyckoff.append(
                atom_orbit_frac_coords[indices_of_projected_points_in_polygon]
            )
            wyckoff_polygon_indices_of_atoms_with_projections_inside.append(
                [i] * len(indices_of_projected_points_in_polygon)
            )

        all_projection_distances = np.concatenate(all_projection_distances, axis=0)
        # (n_points_with_orthogonal_projection_on_wyckoff,)
        points_with_orthogonal_projection_on_wyckoff = np.concatenate(
            points_with_orthogonal_projection_on_wyckoff, axis=0
        )
        # (n_points_with_orthogonal_projection_on_wyckoff, 3)
        wyckoff_polygon_indices_of_atoms_with_projections_inside = np.concatenate(
            wyckoff_polygon_indices_of_atoms_with_projections_inside, axis=0
        )
        # (n_points_with_orthogonal_projection_on_wyckoff,)

        # Of the projected points inside a Wyckoff, choose the one with the
        # shortest distance to the Wyckoff
        if np.min(all_projection_distances) <= 1e-6:
            frac_coord_in_asu = points_with_orthogonal_projection_on_wyckoff[
                all_projection_distances == np.min(all_projection_distances), :
            ][0]  # (3,)
            frac_coord_in_asu = torch.tensor(frac_coord_in_asu, device=device)
            distance_to_asu_wyckoff = np.min(all_projection_distances)
            intersecting_wyckoff_shape_index = int(
                wyckoff_polygon_indices_of_atoms_with_projections_inside[
                    all_projection_distances.argmin()
                ]
            )
        else:
            print(
                f"Minimum distance to Wyckoff polygon was "
                f"{np.min(all_projection_distances)}, larger than the 1e-6 "
                f"threshold. Projecting atoms to Wyckoff polygon edges."
            )
            # -- If no atoms lie inside a Wyckoff facet, then project onto the
            # facet boundaries in the same fashion as for Wyckoff segments
            polygon_edges = []  # (n_segments, 2, 3)
            map_edge_to_polygon = []  # (n_edges,)
            for i, polygon in enumerate(wyckoff_polygons):
                # Get edge segments from polygon
                map_3d_to_2d, _, _ = projection_map_3d_plane_to_2d(polygon[:3, :])
                polygon2d = map_3d_to_2d(polygon)
                assert np.allclose(polygon2d[:, 2], 0.0, atol=1.0e-7)
                polygon2d = polygon2d[:, :2]

                hull = ConvexHull(polygon2d)
                for simplex in hull.simplices:
                    polygon_edges.append(polygon[simplex])
                    map_edge_to_polygon.append(i)
            polygon_edges = np.stack(polygon_edges, axis=0)  # (n_segments, 2, 3)
            (
                frac_coord_in_asu,
                distance_to_asu_wyckoff,
                intersecting_edge_index,
            ) = _get_atom_on_wyckoff_line_segments(
                atom_orbit_frac_coords, polygon_edges, device
            )
            intersecting_wyckoff_shape_index = map_edge_to_polygon[intersecting_edge_index]
            assert frac_coord_in_asu.shape == (3,), f"{frac_coord_in_asu.shape}"
    elif wyckoff_dof == 3:
        # Use is_inside() with an atol parameter. If there are more than 1
        # points inside, arbitrarily choose the first one
        distance_to_asu_wyckoff = 0.0
        asu_hull_equations = global_vars.asu_hull_equations_numpy[space_group_number-1]
        # (n_linear_shape_bounds, 4)
        frac_coord_in_asu = None
        atom_in_asu_found = False
        for tolerance in [0.0, 1.0e-8, 1.0e-6]:
            for i, frac_position in enumerate(atom_orbit_frac_coords):
                if _is_inside(frac_position, asu_hull_equations, epsilon=tolerance):
                    frac_coord_in_asu = torch.tensor(
                        frac_position.astype("float32"), device=device
                    )
                    atom_in_asu_found = True

                if atom_in_asu_found:
                    break
            if atom_in_asu_found:
                break
        intersecting_wyckoff_shape_index = 0
    else:
        raise AttributeError

    if distance_to_asu_wyckoff > 1.0e-6:
        warnings.warn(
            f"Mapping an atom orbit in space group "
            f"{space_group_number} to Wyckoff {wyckoff_letter} "
            f"({wyckoff_dof}-D) with a projection distance of "
            f"{distance_to_asu_wyckoff}. Full orbit was: \n"
            f"{repr(atom_orbit_frac_coords[np.all((atom_orbit_frac_coords >= 0.) & (atom_orbit_frac_coords < 1.), axis=-1).nonzero()])}."
            f"\nChosen coordinate was {frac_coord_in_asu}."
        )
        raise AttributeError

    assert (
        frac_coord_in_asu.shape == (3,)
    ), f"SG {space_group_number}, Wyckoff {wyckoff_letter} ({wyckoff_dof}-D). Shape {frac_coord_in_asu.shape}"
    return frac_coord_in_asu, intersecting_wyckoff_shape_index
    # (3,), int


def _get_atom_on_wyckoff_line_segments(
    atom_orbit_frac_coords: np.array,  # (n_positions, 3)
    line_segment_vertices: np.array,  # (n_segments, 2, 3)
    device: Union[str, torch.device],
) -> Tuple[tensor, tensor, int]:
    # -- Project 'atom_orbit_frac_coords' point P onto each Wyckoff line AB
    atom_orbit_frac_coords = torch.tensor(
        atom_orbit_frac_coords.astype("float32"), device=device
    )
    wyckoff_segments = torch.tensor(
        line_segment_vertices.astype("float32"), device=device
    )  # (n_segments, 2, 3)
    ap = (
        atom_orbit_frac_coords.unsqueeze(1) - wyckoff_segments[:, 0, :]
    )  # (n_positions, n_segments, 3)
    ab = wyckoff_segments[:, 1, :] - wyckoff_segments[:, 0, :]  # (n_segments, 3)
    projection_scalars = (ap * ab.unsqueeze(0)).sum(2) / (ab * ab).sum(1)
    #  (n_positions, n_segments) = (n_positions, n_segments) / (n_segments,)
    projected_coords = wyckoff_segments[:, 0, :] + projection_scalars.unsqueeze(
        -1
    ) * ab.unsqueeze(0)
    # (n_positions, n_segments, 3) = (n_segments, 3) + (n_positions, n_segments, 1) * (1, n_segments, 3)

    # -- Of the coordinates whose projections lie inside the ASU Wyckoff
    #   position, choose that with the smallest distance to the Wyckoff
    projection_distances = torch.linalg.norm(
        projected_coords - atom_orbit_frac_coords.unsqueeze(1), dim=-1
    )
    # (n_positions, n_segments)

    # Coords whose projections lie outside the Wyckoff segments get assigned
    # large projection distances
    projections_inside_wyckoff = torch.logical_and(
        projection_scalars > -1.0e-6, projection_scalars < 1.0 + 1.0e-6
    )  # (n_positions, n_segments)
    assert torch.any(projections_inside_wyckoff), f"{torch.min(projection_scalars)}"
    projection_distances = projection_distances + ~projections_inside_wyckoff * 1.0e10
    # (n_positions, n_segments)

    # # Return fractional coordinate projected onto the ASU Wyckoff segment
    # frac_coord_in_asu = projected_coords[projection_distances == torch.min(projection_distances), :][0]

    # Return the unprojected, original fractional coordinate
    orbit_mask_with_min_distance_to_wyckoff = torch.any(
        projection_distances == torch.min(projection_distances), dim=1
    )  # reduce over segments
    frac_coord_in_asu = atom_orbit_frac_coords[orbit_mask_with_min_distance_to_wyckoff, :][0]
    # (3,)
    intersecting_segment_index = int((
        projection_distances == projection_distances.min()
    ).nonzero(as_tuple=True)[1][0])
    return frac_coord_in_asu, torch.min(projection_distances), intersecting_segment_index


def project_point_to_plane(point, point_on_plane, plane_normal):
    """
    Args:
        point: (n, 3)
        point_on_plane: (1, 3)
        plane_normal: (1, 3)

    Returns:
        projected_points: (n, 3) array of 'points' orthogonally projected onto
            the plane
        projection_distances: (n,) array of distances from 'points' to the plane
    """
    assert (
        len(point.shape) == 2
        and point_on_plane.shape == (1, 3)
        and plane_normal.shape == (1, 3)
    )
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    v = point - point_on_plane  # (n, 3)
    normal_distance = (v * plane_normal).sum(1)[
        :, None
    ]  # numpy broadcasts the rightmost dimensions
    # (n, 1)
    projected_points = point - normal_distance * plane_normal
    return projected_points, np.abs(normal_distance.squeeze(axis=-1))


def projection_map_3d_plane_to_2d(triangle: np.array) -> Tuple[Callable, Callable, np.array]:
    """
    Triangle: shape (n_vertices=3, n_spatial_dims=3)
    See https://math.stackexchange.com/a/4431165
    """
    assert triangle.shape == (3, 3)
    # Translate triangle to intersect the origin
    offset = triangle[0]  # (3,)

    # Get matrix that rotates triangle normal to the z-axis
    p1, p2, p3 = triangle[0, :], triangle[1, :], triangle[2, :]
    new_k = np.cross(p1 - p2, p1 - p3, axis=-1)  # (3,) triangle normal
    new_k = new_k / np.linalg.norm(new_k, axis=-1)  # (3,)
    plane_normal = new_k[None, :]  # (1, 3)

    new_i = p1 - p2
    new_i = new_i / np.linalg.norm(new_i, axis=-1)

    new_j = np.cross(new_k, new_i, axis=-1)  # (3,)

    matrix_3d_to_2d = np.stack((new_i, new_j, new_k), axis=-1)  # (3, 3)
    matrix_2d_to_3d = np.linalg.inv(matrix_3d_to_2d)

    # map_3d_to_2d = lambda x: ((x - offset) @ matrix_3d_to_2d)[:, :2]  # assumes 'x' is (n, 3)
    # map_2d_to_3d = lambda x: \
    #     (np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)) @ matrix_2d_to_3d + offset
    # # assumes 'x' is (n, 2)

    map_3d_to_2d = lambda x: (x - offset) @ matrix_3d_to_2d
    map_2d_to_3d = lambda x: x @ matrix_2d_to_3d + offset
    return map_3d_to_2d, map_2d_to_3d, plane_normal
