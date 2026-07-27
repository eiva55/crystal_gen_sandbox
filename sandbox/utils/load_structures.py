"""Turn each model's raw output files into a uniform List[pymatgen.Structure].

Each generate() implementation still writes files however the underlying
external repo expects (CIF, POSCAR, ...) — this module is the single place
that knows how to read each of those formats back in, so metrics code never
has to care which of the five models produced the input.
"""
from pathlib import Path
from typing import List

from pymatgen.core import Structure


def load_structure_files(directory: str, pattern: str) -> List[Structure]:
    """Load every file matching `pattern` in `directory` (non-recursive).

    pymatgen.Structure.from_file auto-detects CIF/POSCAR/etc. from the
    filename, so one function covers both cases. Files that fail to parse
    are skipped rather than raising, since a single malformed generation
    output shouldn't abort evaluation of the rest.
    """
    structures = []
    for path in sorted(Path(directory).glob(pattern)):
        try:
            structures.append(Structure.from_file(str(path)))
        except Exception as exc:
            print(f"Skipping unparsable structure file {path}: {exc}")
    return structures
