from sandbox.datasets.mp20 import MP20Dataset


def test_loads_inline_cif_from_csv(mini_mp20_csv):
    dataset = MP20Dataset(root=mini_mp20_csv, split="test")
    assert len(dataset) == 2
    assert dataset[0].composition.reduced_formula in ("NaCl", "Al")
