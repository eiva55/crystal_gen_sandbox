"""This module defines `torch.utils.data.Dataset`s for offline learning.
Generally we run a preprocessing script to convert a given dataset of `cif`
represented crystals into our `ASUCrystal` type, and then use the dataset class
in this module (in conjunction with a DataLoader).
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd

from aliases import ndarray, Tensor
from crystal_classes import ASUCrystal, ImmutableASUCrystal, CrystalDict
from utils.io_utils import DATA_DIRECTORY


class AsymmetricUnitDataset(Dataset):
    """Materials Project `20` dataset.

    MP-20 contains 45231 general inorganic materials that differ in both structure
    and composition. There are 89 elements and the materials have 1 - 20 atoms in
    the unit cells. MP-20 includes most experimentally known materials with no more
    than 20 atoms in unit cell.

    Reference: https://github.com/txie-93/cdvae/tree/main/data/mp_20
    """

    def __init__(
        self,
        name: str = "mp_20",
        load_lazily: Optional[bool] = False,
        split: str = "train",
        device: torch.device = "cpu",
    ):
        assert (
            split in ["train", "val", "test"] and
            name in ["mp_20", "mp_20_assumeP1", "mpts_52"]
        )
        if load_lazily:
            raise NotImplementedError

        if "mp_20" in name:
            self.max_atoms = 20
            self.max_elements = 7
        elif name == "mpts_52":
            self.max_atoms = 52
            self.max_elements = 7

        self.device = device
        self.name = name
        self.split = split
        leaf_dir = name
        data_directory: Path = DATA_DIRECTORY / leaf_dir / f"{split}.npz"
        properties_path: Path = DATA_DIRECTORY / leaf_dir / f"{split}_properties.pkl"
        npz: dict = np.load(data_directory)
        properties_df = pd.read_pickle(properties_path)  # TODO: use this for something
        self.indices: ndarray = npz["indices"]
        self.packed: ndarray = npz["packed"]
        flat_crystals: List[ndarray] = np.split(self.packed, self.indices)
        num_crystals = len(flat_crystals)

        self.data: List[ImmutableASUCrystal] = []
        self.space_group_indices = []  # torch.long, shape (n_crystals,)
        self.composition_spaces = []  # torch.float, shape (n_crystals, NUM_ELEMENTS)
        self.lattice_lengths = []  # torch.float, shape (n_crystals, 3)
        self.lattice_angles = []  # torch.float, shape (n_crystals, 3)
        self.n_atoms_per_asu = []  # torch.long, shape (n_crystals,)
        self.padded_element_indices: Tensor = -1 * torch.ones(
            num_crystals, self.max_atoms, dtype=torch.long,
        )  # torch.long, shape (n_crystals, MAX_ATOMS)
        self.padded_wyckoff_indices: Tensor = -1 * torch.ones(
            num_crystals, self.max_atoms, dtype=torch.long,
        )  # torch.long, shape (n_crystals, MAX_ATOMS)
        self.padded_wyckoff_shape_indices: Tensor = -1 * torch.ones(
            num_crystals, self.max_atoms, dtype=torch.long,
        )  # torch.long, shape (n_crystals, MAX_ATOMS)
        self.padded_frac_coords: Tensor = -1.0 * torch.ones(
            num_crystals, self.max_atoms, 3, dtype=torch.float,
        )  # torch.float, shape (n_crystals, MAX_ATOMS, 3)
        self.atoms_mask: Tensor = torch.zeros(
            num_crystals, self.max_atoms, dtype=torch.bool,
        )  # torch.bool, shape (n_crystals, MAX_ATOMS)
        for i, flat in enumerate(flat_crystals):
            crystal: ASUCrystal = ASUCrystal.from_flat(flat)
            num_atoms: int = crystal.num_atoms
            self.data.append(crystal.to_ImmutableASUCrystal())

            self.space_group_indices.append(crystal.space_group_number - 1)
            self.composition_spaces.append(crystal.composition_space)
            self.lattice_lengths.append(crystal.conventional_lattice_lengths)
            self.lattice_angles.append(crystal.conventional_lattice_angles)
            self.n_atoms_per_asu.append(num_atoms)

            # Sort atoms lexicographically by Wyckoff position, then element
            indices: List[int] = list(range(num_atoms))
            sorting_indices = torch.tensor(
                sorted(
                    indices,
                    key=lambda i: (
                        crystal.wyckoff_indices[i], crystal.element_indices[i]
                    )
                ), device=crystal.element_indices.device
            )

            self.padded_element_indices[i, :num_atoms] = crystal.element_indices[sorting_indices]
            self.padded_wyckoff_indices[i, :num_atoms] = crystal.wyckoff_indices[sorting_indices]
            self.padded_wyckoff_shape_indices[i, :num_atoms] = crystal.wyckoff_shape_indices[sorting_indices]
            self.padded_frac_coords[i, :num_atoms, :] = crystal.conventional_frac_coords[sorting_indices]
            self.atoms_mask[i, :num_atoms] = True

        self.space_group_indices = torch.tensor(self.space_group_indices, dtype=torch.long)
        self.n_atoms_per_asu = torch.tensor(self.n_atoms_per_asu, dtype=torch.long)
        self.composition_spaces = torch.stack(self.composition_spaces, dim=0)
        self.lattice_lengths = torch.stack(self.lattice_lengths, dim=0)
        self.lattice_angles = torch.stack(self.lattice_angles, dim=0)

        # For batched sampling from a DataLoader
        self.__getitems__ = self.__getitem__

    def __len__(self) -> int:
        return len(self.space_group_indices)

    def __getitem__(
        self, index: Union[int, List, Tensor]
    ) -> CrystalDict:
        if isinstance(index, int):
            batch_size = 1
        elif isinstance(index, List):
            batch_size = len(index)
            index = torch.tensor(index, dtype=torch.long)
        elif isinstance(index, Tensor):
            assert index.dtype == torch.long
            batch_size = index.numel()
        else:
            raise AttributeError

        crystal_dict = self._get_crystal_dict(index, batch_size)
        return crystal_dict

    @torch.no_grad()
    def _get_crystal_dict(self, index: Union[int, Tensor], batch_size: int) -> CrystalDict:
        crystal_dict = CrystalDict(
            space_group_indices=self.space_group_indices[index].view(batch_size),
            batch_chemistries=self.composition_spaces[index].view(batch_size, -1),  # (batch_size, MAX_ELEMENTS)
            lattice_lengths=self.lattice_lengths[index].view(batch_size, 3),
            lattice_angles=self.lattice_angles[index].view(batch_size, 3),
            n_atoms_per_asu=self.n_atoms_per_asu[index].view(batch_size),
            element_indices=self.padded_element_indices[index].view(batch_size, self.max_atoms),
            wyckoff_indices=self.padded_wyckoff_indices[index].view(batch_size, self.max_atoms),
            wyckoff_shape_indices=self.padded_wyckoff_shape_indices[index].view(batch_size, self.max_atoms),
            frac_coords=self.padded_frac_coords[index].view(batch_size, self.max_atoms, 3),
            atoms_mask=self.atoms_mask[index].view(batch_size, self.max_atoms),
            device=self.space_group_indices.device,
        )
        return crystal_dict

    @torch.no_grad()
    def reindex(self, idxs: Union[List, Tensor] = None):
        if idxs is None:
            # Randomly shuffle dataset
            num_crystals = len(self)
            idxs = torch.randperm(num_crystals)
        else:
            # Slice dataset with specified idxs
            num_crystals = len(idxs)
            if isinstance(idxs, List):
                idxs = torch.tensor(idxs, dtype=torch.long)

        self.data = [self.data[int(i)] for i in idxs]
        self.space_group_indices = self.space_group_indices[idxs].view(-1)
        self.composition_spaces = self.composition_spaces[idxs].view(num_crystals, -1)
        self.lattice_lengths = self.lattice_lengths[idxs].view(num_crystals, -1)
        self.lattice_angles = self.lattice_angles[idxs].view(num_crystals, -1)
        self.n_atoms_per_asu = self.n_atoms_per_asu[idxs].view(num_crystals)
        self.padded_element_indices = self.padded_element_indices[idxs].view(num_crystals, self.max_atoms)
        self.padded_wyckoff_indices = self.padded_wyckoff_indices[idxs].view(num_crystals, self.max_atoms)
        self.padded_wyckoff_shape_indices = self.padded_wyckoff_shape_indices[idxs].view(num_crystals, self.max_atoms)
        self.padded_frac_coords = self.padded_frac_coords[idxs].view(num_crystals, self.max_atoms, 3)
        self.atoms_mask = self.atoms_mask[idxs].view(num_crystals, self.max_atoms)

    @property
    def empirical_space_group_probs(self) -> Tensor:
        counts = torch.bincount(self.space_group_indices, minlength=230)
        return counts / len(self.space_group_indices)
