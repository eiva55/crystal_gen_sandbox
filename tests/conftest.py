import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from pymatgen.core import Lattice, Structure


@pytest.fixture
def synthetic_structures():
    """A couple of small, self-contained structures — no dependency on the
    real (large, machine-specific) MP-20 CSV on disk.
    """
    nacl = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    al_fcc = Structure(
        Lattice.cubic(4.05),
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    return [nacl, al_fcc]


@pytest.fixture
def mini_mp20_csv(tmp_path, synthetic_structures):
    """A tiny fake all.csv, in the same layout as the real MP-20 CSV
    (inline CIF text in a `cif` column), so MP20Dataset can be tested
    without the real ~20k-row file.
    """
    import pandas as pd

    root = tmp_path / "mini_mp20"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)

    rows = [{"material_id": f"fake-{i}", "cif": s.to(fmt="cif")} for i, s in enumerate(synthetic_structures)]
    pd.DataFrame(rows).to_csv(raw_dir / "all.csv", index=False)
    return str(root)
