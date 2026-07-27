from sandbox.metrics.crystal_metrics import CrystalMetrics


def test_validity_counts_non_none_structures(synthetic_structures):
    structures = synthetic_structures + [None]
    assert CrystalMetrics.compute_validity(structures) == 2 / 3


def test_uniqueness_detects_duplicates(synthetic_structures):
    nacl, al_fcc = synthetic_structures
    duplicated = [nacl, nacl, al_fcc]
    uniqueness = CrystalMetrics.compute_uniqueness(duplicated)
    assert uniqueness == pytest.approx(2 / 3)


def test_novelty_is_zero_when_identical_to_reference(synthetic_structures):
    novelty = CrystalMetrics.compute_novelty(synthetic_structures, reference=synthetic_structures)
    assert novelty == 0.0


def test_novelty_is_one_against_unrelated_reference(synthetic_structures):
    nacl, al_fcc = synthetic_structures
    novelty = CrystalMetrics.compute_novelty([nacl], reference=[al_fcc])
    assert novelty == 1.0


import pytest  # noqa: E402 (kept at bottom to mirror pytest.approx usage above)
