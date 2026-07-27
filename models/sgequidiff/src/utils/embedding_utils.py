from pathlib import Path
import warnings
import json

import torch.nn.functional as F

from constants import *
from aliases import *
import global_vars
from utils.io_utils import DATA_DIRECTORY


# EmbeddingTools is to be instantiated once
class EmbeddingTools:
    @torch.no_grad()
    def __init__(
        self,
        space_group_embedding_json_path: str = None,
        element_embedding_json_path: str = None,
        wyckoff_embedding_json_path: str = None,
        chemistry_embedding_type: str = "identity",
        device: Union[str, torch.device] = "cpu",
    ):
        self.space_group_embedding_dict: Union[None, dict] = None
        self.element_embedding_dict: Union[None, dict] = None
        self.wyckoff_embedding_dict: Union[None, dict] = None
        self.chemistry_embedding_type = chemistry_embedding_type
        self.device = device

        # --- Load dictionaries and convert to tensors for batched retrieval
        if space_group_embedding_json_path is not None:
            space_group_embedding_json_path = Path(
                DATA_DIRECTORY / space_group_embedding_json_path
            ).as_posix()
            with open(space_group_embedding_json_path, "r") as file:
                self.space_group_embedding_dict = json.load(file)
            self.space_group_embedding_length = len(self.space_group_embedding_dict["1"])
            self.space_group_embedding_tensor = torch.tensor(
                [self.space_group_embedding_dict[str(sg_num)] for sg_num in range(1, 231)],
                device=self.device,
                dtype=torch.float,
            )  # (230, space_group_embedding_length)
        else:
            self.space_group_embedding_length = NUM_SPACE_GROUPS
        if element_embedding_json_path is not None:
            element_embedding_json_path = Path(
                DATA_DIRECTORY / element_embedding_json_path
            ).as_posix()
            with open(element_embedding_json_path, "r") as file:
                self.element_embedding_dict = json.load(file)
            self.element_embedding_length = len(self.element_embedding_dict["0"])
            self.element_embedding_tensor = torch.tensor(
                [
                    self.element_embedding_dict[str(atomic_number)]
                    for atomic_number in range(NUM_ELEMENTS + 1)
                ],
                device=self.device,
                dtype=torch.float,
            )  # (NUM_ELEMENTS+1, element_embedding_length)
            # NUM_ELEMENTS+1 because one atom type is for the seed atom
        else:
            self.element_embedding_length = NUM_ELEMENTS
        if wyckoff_embedding_json_path is not None:
            wyckoff_embedding_json_path = Path(
                DATA_DIRECTORY / wyckoff_embedding_json_path
            ).as_posix()
            with open(wyckoff_embedding_json_path, "r") as file:
                self.wyckoff_embedding_dict = json.load(file)
            self.wyckoff_embedding_length = len(self.wyckoff_embedding_dict["1"]["a"])

            self.wyckoff_embedding_tensor = []
            self.n_wyckoffs_per_space_group = []
            for sg_num in range(1, 231):
                wyckoff_dict_of_sg_num = self.wyckoff_embedding_dict[str(sg_num)]
                wyckoff_letters_of_sg_num = list(wyckoff_dict_of_sg_num.keys())

                # Sort Wyckoff letters from high to low symmetry (note space
                # group 47 has 27 Wyckoff positions, so we handle uppercase
                # letters too)
                wyckoff_ascii_integers = [ord(letter) for letter in wyckoff_letters_of_sg_num]
                wyckoff_indices = [
                    ascii_int - 97 if ascii_int >= 97 else ascii_int - 65 + 26
                    for ascii_int in wyckoff_ascii_integers
                ]
                sorted_wyckoff_letters_of_sg_num = [
                    letter
                    for letter, _ in sorted(
                        zip(wyckoff_letters_of_sg_num, wyckoff_indices),
                        key=lambda pair: pair[1],
                    )
                ]

                # Construct (MAX_WYCKOFF_SITES, wyckoff_embedding_length) tensor.
                # Pad with zeroes since different space groups have different
                # numbers of Wyckoff positions
                sorted_wyckoff_embeddings_of_sg_num = torch.tensor(
                    [
                        wyckoff_dict_of_sg_num[letter]
                        for letter in sorted_wyckoff_letters_of_sg_num
                    ],
                    device=self.device,
                    dtype=torch.float,
                )  # (n_wyckoffs_in_sg_num, wyckoff_embedding_length)
                zero_padding = torch.zeros(
                    (
                        MAX_WYCKOFF_SITES - len(wyckoff_letters_of_sg_num),
                        self.wyckoff_embedding_length,
                    ),
                    device=self.device,
                )
                self.wyckoff_embedding_tensor.append(
                    torch.cat([sorted_wyckoff_embeddings_of_sg_num, zero_padding], dim=0)
                )
                self.n_wyckoffs_per_space_group.append(len(wyckoff_letters_of_sg_num))
            self.wyckoff_embedding_tensor = torch.stack(self.wyckoff_embedding_tensor, dim=0)
            # (NUM_SPACEGROUPS, MAX_WYCKOFF_SITES, wyckoff_embedding_length)
            self.n_wyckoffs_per_space_group = torch.tensor(
                self.n_wyckoffs_per_space_group, device=self.device, dtype=torch.long
            )
            # (NUM_SPACEGROUPS,)
        else:
            self.wyckoff_embedding_length = MAX_WYCKOFF_SITES

    def get_space_group_embedding(self, space_group_index: tensor) -> tensor:
        """
        Args:
            space_group_index: shape (batch_size,) LongTensor of zero-indexed space
                group indices
        Returns:
            shape (batch_size, space_group_embedding_length) tensor
        """
        assert space_group_index.dtype == torch.long
        if self.space_group_embedding_dict is None:
            return F.one_hot(space_group_index, NUM_SPACE_GROUPS).float()
        else:
            return self.space_group_embedding_tensor.to(space_group_index.device)[
                space_group_index
            ]

    @torch.no_grad()
    def get_element_embedding(self, atomic_number: tensor) -> tensor:
        """
        Args:
            atomic_number: (batch_size,) LongTensor of atomic numbers in
                [1, NUM_ELEMENTS]. Zero is reserved for null atoms.
        Returns:
            (batch_size, element_embedding_length) tensor
        """
        assert atomic_number.dtype == torch.long
        if self.element_embedding_dict is None:
            return F.one_hot(atomic_number - 1, NUM_ELEMENTS).float()
        else:
            return self.element_embedding_tensor.to(atomic_number.device)[atomic_number]

    @torch.no_grad()
    def get_wyckoff_embedding(
        self, wyckoff_index: tensor, space_group_index: tensor
    ) -> tensor:
        """
        Args:
            wyckoff_index: shape (batch_size,) LongTensor of indices from 0 to
                MAX_WYCKOFF_SITES-1 corresponding to alphabetically ordered
                Wyckoff letters (high to low symmetry, or small to large orbits)
            space_group_index: shape (batch_size,) LongTensor of 0-indexed
                space groups in the corresponding order as 'wyckoff_index'.
        Returns:
            (batch_size, wyckoff_embedding_length)
        """
        if self.wyckoff_embedding_dict is None:
            return F.one_hot(wyckoff_index, MAX_WYCKOFF_SITES).float()
        else:
            self.n_wyckoffs_per_space_group = self.n_wyckoffs_per_space_group.to(
                space_group_index.device
            )
            # Check the Wyckoff and space group index pairs are valid
            valid_wyckoff_space_group_pairs: tensor = (
                self.n_wyckoffs_per_space_group[space_group_index] > wyckoff_index
            )
            assert valid_wyckoff_space_group_pairs.all(), (
                f"Invalid space group-Wyckoff index pairs for space group "
                f"indices {space_group_index[torch.where(~valid_wyckoff_space_group_pairs)[0]]} "
                f"and Wyckoff indices {wyckoff_index[torch.where(~valid_wyckoff_space_group_pairs)[0]]}."
            )
            return self.wyckoff_embedding_tensor.to(wyckoff_index.device)[
                space_group_index, wyckoff_index, :
            ]  # (batch_size, wyckoff_embedding_length)

    def get_chemistry_embedding(self, chemistry: tensor) -> tensor:
        if self.chemistry_embedding_type == "identity":
            return chemistry
        else:
            # Todo: DeepSets to process chemistries?
            raise ValueError("chemistry_embedding_type value is invalid")


def set_global_embedding_tools(
    space_group_embedding_json_path: str = None,
    element_embedding_json_path: str = None,
    wyckoff_embedding_json_path: str = None,
    chemistry_embedding_type: str = "identity",
    device: Union[str, torch.device] = "cpu",
):
    global_vars.embedding_tools = EmbeddingTools(
        space_group_embedding_json_path=space_group_embedding_json_path,
        element_embedding_json_path=element_embedding_json_path,
        wyckoff_embedding_json_path=wyckoff_embedding_json_path,
        chemistry_embedding_type=chemistry_embedding_type,
        device=device,
    )
