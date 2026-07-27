import pytest
from pymatgen.core import Structure
from sandbox.metrics.crystal_metrics import CrystalMetrics

def test_validity():
    structures = [Structure.from_spacegroup("Fm-3m", [[1,0,0],[0,1,0],[0,0,1]], ["Na", "Cl"], [[0,0,0],[0.5,0.5,0.5]])]
    metrics = CrystalMetrics.compute_validity(structures)
    assert metrics == 1.0

def test_uniqueness():
    s1 = Structure.from_spacegroup("Fm-3m", [[1,0,0],[0,1,0],[0,0,1]], ["Na", "Cl"], [[0,0,0],[0.5,0.5,0.5]])
    s2 = s1.copy()
    metrics = CrystalMetrics.compute_uniqueness([s1, s2])
    assert metrics == 0.5
