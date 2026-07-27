import unittest
import os

import pandas as pd
import numpy as np
import matminer.datasets
from pyxtal.util import symmetrize
from spglib import get_symmetry_dataset
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from utils.data_utils import asu_to_pymatgen_structure, pyxtal_pymatgen_structure_to_asu
from utils.io_utils import get_project_dir
from aliases import *


class TestSpglibConventionalToASUandBack(unittest.TestCase):
    def test(self):
        df = self.load_structure_for_every_space_group()
        test_failed = False  # init
        for i in range(len(df.index)):
            row = df.iloc[i]
            space_group = row["space_group"]
            hall_number = row["hall_number"]
            print(f"----- Attempting space group {space_group}")

            # -- Get pyxtal/cctbx conventional structure
            original_structure = row["pyxtal_conventional_structure"]

            # -- Convert to ASU
            asu = pyxtal_pymatgen_structure_to_asu(original_structure)

            # -- Convert back to conventional structure
            new_structure = asu_to_pymatgen_structure(asu)

            structures_are_close, msg = self.check_structures_are_close(
                original_structure, new_structure
            )
            if not structures_are_close:
                test_failed = True
                print(
                    f"Mismatch for space group {space_group}, "
                    f"pyxtal hall number {hall_number}. "
                )
                print(msg)

        if test_failed:
            raise ValueError

    def check_structures_are_close(
        self, structure1: pmg_structure, structure2: pmg_structure
    ) -> Tuple[bool, str]:
        structures_are_close = True
        msg = ""
        if len(structure1) != len(structure2):
            structures_are_close = False
            msg += (
                f"Original structure had {len(structure1)} sites, but new"
                f" structure has {len(structure2)} sites.\n"
            )
        if structure1.lattice.find_mapping(structure2.lattice) is None:
            structures_are_close = False
            msg += (
                f"Original structure had lattice parameters: {structure1.lattice.parameters}\n"
                f"New structure has lattice parameters {structure2.lattice.parameters}\n"
            )
        for site1 in structure1:
            site_found = any(
                [
                    site1.species == site2.species
                    and site1.properties == site2.properties
                    and np.allclose(site1.frac_coords, site2.frac_coords, atol=1e-6)
                    for site2 in structure2
                ]
            )
            if not site_found and (len(structure1) == len(structure2)):
                structures_are_close = False
                msg += f"Original structure has site {site1}, but new structure does not.\n"
        return structures_are_close, msg

    def load_structure_for_every_space_group(self):
        filepath = get_project_dir() + "/test/pyxtal_symmetrized_structure_per_space_group.pkl"
        if not os.path.exists(filepath):
            df = matminer.datasets.load_dataset("matbench_mp_e_form")
            df = df.apply(get_space_group, axis=1)

            example_crystals = []
            missing_space_groups = []
            for space_group in range(1, 231):
                try:
                    row = df[df["space_group"] == space_group].iloc[0].to_frame().transpose()
                    example_crystals.append(row)
                except:
                    missing_space_groups.append(space_group)
            df_with_1_structure_per_space_group = pd.concat(
                example_crystals, ignore_index=True, axis=0
            )
            df_with_1_structure_per_space_group.to_pickle(filepath)
            print(
                f"Dataset did not contain the following space groups:\n {missing_space_groups}"
            )
        else:
            df_with_1_structure_per_space_group = pd.read_pickle(filepath)
        return df_with_1_structure_per_space_group


def get_space_group(row):
    structure = row["structure"]
    pyxtal_conventional_structure, hall_number = symmetrize(
        structure, tol=0.1, a_tol=5.0, style="pyxtal"
    )
    sga = SpacegroupAnalyzer(pyxtal_conventional_structure, symprec=0.1, angle_tolerance=5.0)
    if hall_number != sga._space_group_data["hall_number"]:
        # Overwrite SpacegroupAnalyzer attributes based on spglib default Hall
        # number with attributes based on pyxtal/cctbx/ITA default Hall number
        sga._space_group_data = get_symmetry_dataset(
            sga._cell, 0.1, angle_tolerance=5.0, hall_number=hall_number
        )
    row["pyxtal_conventional_structure"] = pyxtal_conventional_structure
    row["hall_number"]: int = hall_number
    row["space_group"] = sga.get_space_group_number()
    return row
