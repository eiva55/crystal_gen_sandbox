"""Find single-element reference structures in the MP-20 CSV.

PhaseDiagram construction requires one terminal (single-element) entry per
element appearing in the structures being evaluated. Pure MP-20 is a
metastable-compounds dataset, so single-element rows are rare — scanning the
`elements` column (already present in the CSV, no CIF parsing needed) to find
them directly is far cheaper than hoping random sampling stumbles onto them.
"""
import ast
import os
from typing import List, Set

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser


def find_elemental_structures(csv_path: str, needed_elements: Set[str]) -> List[Structure]:
    df = pd.read_csv(csv_path)
    structures = []
    found_elements = set()

    for _, row in df.iterrows():
        try:
            elements = ast.literal_eval(row["elements"])
        except Exception:
            continue
        if len(elements) != 1:
            continue
        element = elements[0]
        if element not in needed_elements or element in found_elements:
            continue
        try:
            structure = CifParser.from_str(row["cif"]).parse_structures(primitive=True)[0]
        except Exception:
            continue
        structures.append(structure)
        found_elements.add(element)

    missing = needed_elements - found_elements
    if missing:
        print(f"No elemental reference found in MP-20 for: {sorted(missing)}")
    return structures
