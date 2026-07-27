import json
from typing import List

import pyxtal
from pyxtal.symmetry import Group

from utils.io_utils import DATA_DIRECTORY
from aliases import *

assert pyxtal.__version__ == "0.6.7", "SymmCD requires a different version of pyxtal (0.6.7 instead of 0.6.0)."


def main():
    # Wyckoff features have 15x13=195 dimensions
    wyckoff_position_embedding_dict = {}
    for sg_number in range(1, 231):
        wp_embedding_dict = {}
        group = Group(sg_number)
        wps = group.Wyckoff_positions  # sorted by descending multiplicity
        for wp in wps:
            wyckoff_letter: str = wp.letter
            wp.get_site_symmetry()  # initialize the wyckoff position?
            site_symm_binarys = wp.get_site_symmetry_object().to_one_hot()
            # (num_sym_axes=15, num_sym_ops=13)
            wp_embedding_dict[wyckoff_letter]: List[float] = (
                site_symm_binarys.astype("float32").flatten().tolist()
            )
        wyckoff_position_embedding_dict[str(sg_number)] = wp_embedding_dict

    filepath: Path = DATA_DIRECTORY / "init_tokens/wyckoff_features/symmcd_wyckoff_features.json"
    with open(filepath, "w") as file:
        json.dump(wyckoff_position_embedding_dict, file)


if __name__ == "__main__":
    main()
