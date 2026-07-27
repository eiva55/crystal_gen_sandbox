import json
from typing import List

import pyxtal
from pyxtal.symmetry import Group

from utils.io_utils import DATA_DIRECTORY
from aliases import *

assert pyxtal.__version__ == "0.6.7", "SymmCD requires a different version of pyxtal (0.6.7 instead of 0.6.0)."


LATTICE_MAPPER = {"P": 0, "I": 1, "F": 2, "A": 3, "B": 4, "C": 5, "R": 6}


def get_spacegroup_binary_repr(number:int):
    spg = Group(number)

    # get the point group and translation representation of space group
    ss = spg.get_spg_symmetry_object()
    axis_wise_binary_repr = torch.from_numpy(ss.to_matrix_representation_spg().reshape(-1,))

    # join the bravais lattice type
    lattice_type = LATTICE_MAPPER[spg.symbol[0]]
    lattice_type_repr = torch.zeros(7)
    lattice_type_repr[lattice_type] = 1
    return torch.cat([lattice_type_repr, axis_wise_binary_repr], dim=0)


def main():
    # Space group features have 397 dimensions
    space_group_embedding_dict = {}
    for sg_number in range(1, 231):
        space_group_embedding_dict[str(sg_number)]: List[float] = (
            get_spacegroup_binary_repr(sg_number).tolist()
        )

    filepath: Path = DATA_DIRECTORY / "init_tokens/wyckoff_features/symmcd_space_group_features.json"
    with open(filepath, "w") as file:
        json.dump(space_group_embedding_dict, file)


if __name__ == "__main__":
    main()
