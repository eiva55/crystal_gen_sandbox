from dataclasses import dataclass

import numpy as np

from aliases import *
from constants import *
import global_vars


@dataclass
class CartesianAtom:
    wyckoff_letter: Tensor  # zero-indexed
    element: Tensor  # zero-indexed
    cartesian_cart_coords: Tensor  # shape (1, 3)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CartesianAtom):
            return NotImplemented

        return (
            self.wyckoff_letter == other.wyckoff_letter
            and self.element == other.element
            and torch.allclose(
                self.cartesian_cart_coords,
                other.cartesian_cart_coords,
                atol=0.1,
                rtol=0.0,
            )
        )

    def __str__(self) -> str:
        """Unique string identifier. Used for hashing ASUCrystal objects"""
        string_id = (
            f"{int(self.wyckoff_letter)}_{int(self.element)}_"
            f"{self.cartesian_cart_coords.detach().cpu().round(decimals=1)}"
        )
        return string_id


@dataclass
class ASUCrystal:
    """
    Dataclass containing raw attributes of a crystal.

    Instance Attributes
    ----------
    space_group_number: Tensor (torch.long)
        Shape (,). 1-indexed space group number from [1-230]
    conventional_lattice_lengths: Tensor (torch.float)
        Shape (3,). Lengths (a, b, c)
    conventional_lattice_angles: Tensor (torch.float)
        Shape (3,). Angles (alpha, beta, gamma)
    element_indices: Tensor (torch.long)
        Shape (n_inequivalent_atoms,). Zero-indexed atom types
    wyckoff_indices: Tensor (torch.long)
        Shape (n_inequivalent_atoms,). Zero-indexed wyckoff positions
    conventional_frac_coords: Tensor (torch.float)
        Shape (n_inequivalent_atoms, 3). Fractional coordinates for
        atoms inside the asymmetric unit.
    wyckoff_shape_indices: Tensor (torch.long)
        Shape (n_inequivalent_atoms,). Zero-indexed shapes in a given Wyckoff
        position, in the same order as in
            global_vars.asu_wyckoff_dict[space_group_index][wyckoff_letter]["vertices"],
        that each atom lies on.
    composition_space: Tensor (torch.float)
        shape (NUM_ELEMENTS,) one-hot encoding of the composition space, where a
        1.0 indicates the presence of an element in the composition space.
    """

    # Static variables
    __hash__ = None

    def __init__(
        self,
        space_group_number: tensor,
        conventional_lattice_lengths: tensor,
        conventional_lattice_angles: tensor,
        element_indices: tensor,
        wyckoff_indices: tensor,
        conventional_frac_coords: tensor,
        device: Union[str, torch.device] = "cpu",
        wyckoff_shape_indices: Optional[Tensor] = None,
        composition_space: Optional[Tensor] = None,
        cartesian_coords: Optional[Tensor] = None,
    ):
        assert (
            wyckoff_indices.shape[0]
            == element_indices.shape[0]
            == conventional_frac_coords.shape[0]
        )
        self.space_group_number = space_group_number
        self.conventional_lattice_lengths = conventional_lattice_lengths
        self.conventional_lattice_angles = conventional_lattice_angles
        self.element_indices = element_indices
        self.wyckoff_indices = wyckoff_indices
        self.conventional_frac_coords = conventional_frac_coords
        self.device = device

        # Optional kwargs
        self.wyckoff_shape_indices = wyckoff_shape_indices
        self.cartesian_coords: Optional[Tensor] = cartesian_coords
        self.composition_space = composition_space

    @classmethod
    def from_flat(cls, flat_crystal: ndarray):
        num_atoms: int = int(flat_crystal[0])
        space_group_number: Tensor = torch.tensor(flat_crystal[1].astype("int64"), dtype=torch.long)
        composition_space: Tensor = torch.tensor(
            flat_crystal[2 : 2 + NUM_ELEMENTS], dtype=torch.float
        )
        conventional_lattice_lengths: Tensor = torch.tensor(
            flat_crystal[2 + NUM_ELEMENTS : 5 + NUM_ELEMENTS], dtype=torch.float
        )
        conventional_lattice_angles: Tensor = torch.tensor(
            flat_crystal[5 + NUM_ELEMENTS : 8 + NUM_ELEMENTS], dtype=torch.float
        )
        element_indices: Tensor = torch.tensor(
            flat_crystal[8 + NUM_ELEMENTS : 8 + NUM_ELEMENTS + num_atoms], dtype=torch.long
        )
        wyckoff_indices: Tensor = torch.tensor(
            flat_crystal[8 + NUM_ELEMENTS + num_atoms : 8 + NUM_ELEMENTS + (2 * num_atoms)],
            dtype=torch.long,
        )
        conventional_frac_coords: Tensor = torch.tensor(
            flat_crystal[8 + NUM_ELEMENTS + (2 * num_atoms) : 8 + NUM_ELEMENTS + (5 * num_atoms)].reshape(num_atoms, 3),
            dtype=torch.float,
        )
        if len(flat_crystal) > 8 + NUM_ELEMENTS + (5 * num_atoms):
            wyckoff_shape_indices: Tensor = torch.tensor(
                flat_crystal[8 + NUM_ELEMENTS + (5 * num_atoms) : 8 + NUM_ELEMENTS + (6 * num_atoms)],
                dtype=torch.long
            )
        else:
            wyckoff_shape_indices = None

        return ASUCrystal(
            space_group_number=space_group_number,
            composition_space=composition_space,
            conventional_lattice_lengths=conventional_lattice_lengths,
            conventional_lattice_angles=conventional_lattice_angles,
            element_indices=element_indices,
            wyckoff_indices=wyckoff_indices,
            conventional_frac_coords=conventional_frac_coords,
            wyckoff_shape_indices=wyckoff_shape_indices,
        )

    @property
    def num_atoms(self) -> int:
        return int(self.conventional_frac_coords.shape[0])

    def flatten(self) -> ndarray:
        if self.composition_space is None:
            composition_space = torch.zeros(NUM_ELEMENTS, dtype=torch.float, device=self.device)
            composition_space[self.element_indices] = 1.0
        else:
            composition_space = self.composition_space

        flat_crystal = [
            np.array([float(len(self.element_indices))]),
            np.array([float(self.space_group_number)]),
            composition_space.cpu().numpy(),    # optional
            self.conventional_lattice_lengths.cpu().numpy(),
            self.conventional_lattice_angles.cpu().numpy(),
            self.element_indices.cpu().numpy(),
            self.wyckoff_indices.cpu().numpy(),
            self.conventional_frac_coords.cpu().numpy().ravel(),
        ]
        if self.wyckoff_shape_indices is not None:
            flat_crystal.append(self.wyckoff_shape_indices.cpu().numpy().ravel())

        flat_crystal = np.concatenate(flat_crystal)
        return flat_crystal

    def to(self, device: Union[str, torch.device] = "cpu"):
        """
        If 'device' matches the device of the tensors in 'self', then return
        'self'. Else, return a copy of 'self' with tensors on 'device'.
        """
        if isinstance(device, str):
            device = torch.device(type=device)
        if self.element_indices.device.type == device.type:
            return self

        if isinstance(self.space_group_number, tensor):
            space_group_number = self.space_group_number.to(device)
        elif isinstance(self.space_group_number, int):
            space_group_number = self.space_group_number
        else:
            raise AttributeError

        if isinstance(self.cartesian_coords, Tensor):
            cartesian_coords = self.cartesian_coords.to(device)
        else:
            cartesian_coords = self.cartesian_coords

        if isinstance(self.composition_space, Tensor):
            composition_space = self.composition_space.to(device)
        else:
            composition_space = self.composition_space

        if isinstance(self.wyckoff_shape_indices, Tensor):
            wyckoff_shape_indices = self.wyckoff_shape_indices.to(device)
        else:
            wyckoff_shape_indices = self.wyckoff_shape_indices

        new_device_asu = ASUCrystal(
            space_group_number=space_group_number,  # Index this from 1-230
            conventional_lattice_lengths=self.conventional_lattice_lengths.to(device),
            conventional_lattice_angles=self.conventional_lattice_angles.to(device),
            element_indices=self.element_indices.to(device),
            wyckoff_indices=self.wyckoff_indices.to(device),
            conventional_frac_coords=self.conventional_frac_coords.to(device),
            wyckoff_shape_indices=wyckoff_shape_indices,
            composition_space=composition_space,
            cartesian_coords=cartesian_coords,
            device=device,
        )
        return new_device_asu

    def detach(self):
        if isinstance(self.space_group_number, tensor):
            space_group_number = self.space_group_number.detach()
        elif isinstance(self.space_group_number, int):
            space_group_number = self.space_group_number
        else:
            raise AttributeError

        if isinstance(self.cartesian_coords, Tensor):
            cartesian_coords = self.cartesian_coords.detach()
        else:
            cartesian_coords = self.cartesian_coords

        if isinstance(self.composition_space, Tensor):
            composition_space = self.composition_space.detach()
        else:
            composition_space = self.composition_space

        if isinstance(self.wyckoff_shape_indices, Tensor):
            wyckoff_shape_indices = self.wyckoff_shape_indices.detach()
        else:
            wyckoff_shape_indices = self.wyckoff_shape_indices

        detached_asu = ASUCrystal(
            space_group_number=space_group_number,  # Index this from 1-230
            conventional_lattice_lengths=self.conventional_lattice_lengths.detach(),
            conventional_lattice_angles=self.conventional_lattice_angles.detach(),
            element_indices=self.element_indices.detach(),
            wyckoff_indices=self.wyckoff_indices.detach(),
            conventional_frac_coords=self.conventional_frac_coords.detach(),
            composition_space=composition_space,
            wyckoff_shape_indices=wyckoff_shape_indices,
            cartesian_coords=cartesian_coords,
            device=self.device,
        )
        return detached_asu

    def get_atom_info(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.wyckoff_indices[idx],
            self.element_indices[idx],
            self.conventional_frac_coords[idx],
            self.wyckoff_shape_indices[idx] if self.wyckoff_shape_indices is not None else None,
        )

    def delete_atom(self, idx: int):
        assert idx >= 0, "Negative indices not supported."
        self.element_indices = torch.cat([self.element_indices[:idx], self.element_indices[idx + 1 :]], dim=0)
        self.wyckoff_indices = torch.cat(
            [self.wyckoff_indices[:idx], self.wyckoff_indices[idx + 1 :]], dim=0
        )
        self.conventional_frac_coords = torch.cat(
            [
                self.conventional_frac_coords[:idx],
                self.conventional_frac_coords[idx + 1 :],
            ],
            dim=0,
        )
        if isinstance(self.wyckoff_shape_indices, Tensor):
            self.wyckoff_shape_indices = torch.cat(
                [
                    self.wyckoff_shape_indices[:idx],
                    self.wyckoff_shape_indices[idx + 1:],
                ],
                dim=0,
            )

    def to_ImmutableASUCrystal(self):
        return ImmutableASUCrystal(
            self.space_group_number,
            self.conventional_lattice_lengths,
            self.conventional_lattice_angles,
            self.element_indices,
            self.wyckoff_indices,
            self.conventional_frac_coords,
            self.device,
            self.wyckoff_shape_indices,
            self.composition_space,
            self.cartesian_coords,
        )

    def __str__(self):
        # element_indices = self.composition_space.nonzero().view(-1).tolist()
        # elements = ", ".join([chemical_symbols[i + 1] for i in element_indices])
        str_representation = (
            f"----- ASUCrystal -----\n"
            f"Space group {int(self.space_group_number)}\n"
            # f"Elements {elements}\n"
            f"(a={self.conventional_lattice_lengths[0].round(decimals=2):0.2f},"
            f"b={self.conventional_lattice_lengths[1].round(decimals=2):0.2f},"
            f"c={self.conventional_lattice_lengths[2].round(decimals=2):0.2f},"
            f"alpha={self.conventional_lattice_angles[0].round():0.0f},"
            f"beta={self.conventional_lattice_angles[1].round():0.0f},"
            f"gamma={self.conventional_lattice_angles[2].round():0.0f})\n"
            f"-- Atoms:\n"
        )
        wyckoff_letters = [
            chr(97 + self.wyckoff_indices[i])
            if self.wyckoff_indices[i] <= 25
            else chr(39 + self.wyckoff_indices[i])
            for i in range(self.num_atoms)
        ]
        wyckoff_dims = [
            str(
                global_vars.asu_wyckoff_dict[str(self.space_group_number.item())][letter][
                    "dim"
                ]
            )
            for letter in wyckoff_letters
        ]
        atom_strings = "\n".join(
            [
                "".join(
                    [
                        chemical_symbols[int(self.element_indices[i]) + 1],
                        "\t",
                        wyckoff_letters[i] + " (" + wyckoff_dims[i] + "D)",
                        "\t",
                        str(self.conventional_frac_coords[i].tolist()),
                    ]
                )
                for i in range(self.num_atoms)
            ]
        )
        str_representation = "".join((str_representation, atom_strings, "\n"))
        return str_representation


class ImmutableASUCrystal:
    def __init__(
        self,
        space_group_number: tensor,
        conventional_lattice_lengths: tensor,
        conventional_lattice_angles: tensor,
        element_indices: tensor,
        wyckoff_indices: tensor,
        conventional_frac_coords: tensor,
        device: Union[str, torch.device] = "cpu",
        wyckoff_shape_indices: Optional[Tensor] = None,
        composition_space: Optional[Tensor] = None,
        cartesian_coords: Optional[Tensor] = None,
    ):
        assert (
            wyckoff_indices.shape[0]
            == element_indices.shape[0]
            == conventional_frac_coords.shape[0]
        )
        self._space_group_number = space_group_number
        self._conventional_lattice_lengths = conventional_lattice_lengths
        self._conventional_lattice_angles = conventional_lattice_angles
        self._element_indices = element_indices
        self._wyckoff_indices = wyckoff_indices
        self._conventional_frac_coords = conventional_frac_coords
        self._device = device
        self._num_atoms = int(self._conventional_frac_coords.shape[0])

        self._wyckoff_shape_indices = wyckoff_shape_indices
        self._composition_space = composition_space
        self._cartesian_coords: Union[Tensor, None] = cartesian_coords

        # Wrap atom information into objects for hashing
        if isinstance(cartesian_coords, Tensor):
            self._atoms: List[CartesianAtom] = [
                CartesianAtom(wyckoff, type, cart_coord.unsqueeze(0))
                for i, (wyckoff, type, cart_coord) in enumerate(
                    zip(
                        wyckoff_indices.detach(),
                        element_indices.detach(),
                        cartesian_coords.detach(),
                    )
                )
            ]
        else:
            self._atoms = None

    def to_ASUCrystal(self):
        return ASUCrystal(
            self._space_group_number.clone(),
            self._conventional_lattice_lengths.clone(),
            self._conventional_lattice_angles.clone(),
            self._element_indices.clone(),
            self._wyckoff_indices.clone(),
            self._conventional_frac_coords.clone(),
            self._device,
            self._wyckoff_shape_indices.clone() if self._wyckoff_shape_indices is not None else None,
            self._composition_space.clone() if self._composition_space is not None else None,
            self._cartesian_coords,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImmutableASUCrystal):
            return NotImplemented

        # Don't compare composition spaces?
        if other is self:
            return True
        if self._space_group_number != other._space_group_number:
            return False
        if self._num_atoms != other._num_atoms:
            return False
        if not (
            torch.allclose(
                self._conventional_lattice_lengths,
                other._conventional_lattice_lengths,
                atol=1e-6,
                rtol=0.0,
            )
            and torch.allclose(
                self._conventional_lattice_angles,
                other._conventional_lattice_angles,
                atol=1e-6,
                rtol=0.0,
            )
        ):
            return False

        assert self._atoms is not None
        return all(atom in other._atoms for atom in self._atoms) and all(
            atom in self._atoms for atom in other._atoms
        )

    def __hash__(self) -> str:
        assert self._atoms is not None
        # Enables O(1) lookup times in dict or set
        crystal_str = (
            f"{int(self._space_group_number)}_"
            f"{self._conventional_lattice_lengths.detach().cpu().round(decimals=1)}_"
            f"{self._conventional_lattice_angles.detach().cpu().round(decimals=0)}_"
            f"{sorted([str(atom) for atom in self._atoms])}"
        )
        return crystal_str


class CrystalDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def repeat_(self, num_repeats: int):
        """
        Repeat crystals in the batch. For training on multiple trajectories
        per crystal. Repeats must be consecutive for compatibility with
        src.custom_loss.mle_loss_for_learned_pb().
        """
        for key, crystal_attribute_tensor in self.items():
            if isinstance(crystal_attribute_tensor, Tensor):
                self[key] = torch.repeat_interleave(
                    crystal_attribute_tensor, num_repeats, dim=0
                )
        return self

    def __len__(self) -> int:
        return self["space_group_indices"].shape[0]

    def to_(self, device: torch.device):
        self["device"] = device
        self["space_group_indices"] = self["space_group_indices"].to(
            device, non_blocking=True, copy=True
        )
        self["batch_chemistries"] = self["batch_chemistries"].to(
            device, non_blocking=True, copy=True
        )
        self["lattice_lengths"] = self["lattice_lengths"].to(
            device, non_blocking=True, copy=True
        )
        self["lattice_angles"] = self["lattice_angles"].to(
            device, non_blocking=True, copy=True
        )
        if "lattice_matrices" in self:
            self["lattice_matrices"] = self["lattice_matrices"].to(
            device, non_blocking=True, copy=True
        )
        self["n_atoms_per_asu"] = self["n_atoms_per_asu"].to(
            device, non_blocking=True, copy=True
        )
        self["element_indices"] = self["element_indices"].to(
            device, non_blocking=True, copy=True
        )
        self["wyckoff_indices"] = self["wyckoff_indices"].to(
            device, non_blocking=True, copy=True
        )
        self["wyckoff_shape_indices"] = self["wyckoff_shape_indices"].to(
            device, non_blocking=True, copy=True
        )
        self["frac_coords"] = self["frac_coords"].to(
            device, non_blocking=True, copy=True
        )
        self["atoms_mask"] = self["atoms_mask"].to(
            device, non_blocking=True, copy=True
        )
        if "wyckoff_2d_simplex_coords" in self:
            self["wyckoff_2d_simplex_coords"] = self["wyckoff_2d_simplex_coords"].to(
                device, non_blocking=True, copy=True
            )
        return self

    def get_asu_crystal_objects(self) -> List[ASUCrystal]:
        crystals: List[ASUCrystal] = []
        for i in range(self["n_atoms_per_asu"].shape[0]):
            mask: Tensor = self["atoms_mask"][i]
            # (MAX_ATOMS,)
            crystals.append(
                ASUCrystal(
                    space_group_number=1 + self["space_group_indices"][i],
                    # Index from 1-230
                    composition_space=self["batch_chemistries"][i],
                    conventional_lattice_lengths=self["lattice_lengths"][i],
                    conventional_lattice_angles=self["lattice_angles"][i],
                    element_indices=self["element_indices"][i][mask],
                    wyckoff_indices=self["wyckoff_indices"][i][mask],
                    conventional_frac_coords=self["frac_coords"][i][mask],
                    device=self["device"],
                    wyckoff_shape_indices=self["wyckoff_shape_indices"][i][mask],
                )
            )
        return crystals
