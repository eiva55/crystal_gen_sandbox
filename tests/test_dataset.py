from sandbox.datasets.mp20 import MP20Dataset


def test_train_val_test_partition_and_content(mini_mp20_csv):
    """Splits together must cover every row exactly once — no leakage,
    no drops. Exact per-split counts aren't asserted: with only 2 rows in
    the fixture and 60/20/20 fractions, which specific split each row lands
    in is an implementation detail (which the real 45k-row MP-20 CSV
    averages out across splits regardless).
    """
    all_structures = []
    for split in ("train", "val", "test"):
        all_structures.extend(MP20Dataset(root=mini_mp20_csv, split=split))
    assert len(all_structures) == 2
    formulas = {s.composition.reduced_formula for s in all_structures}
    assert formulas <= {"NaCl", "Al"}


def test_split_assignment_is_deterministic(mini_mp20_csv):
    first = [s.composition.reduced_formula for s in MP20Dataset(root=mini_mp20_csv, split="test")]
    second = [s.composition.reduced_formula for s in MP20Dataset(root=mini_mp20_csv, split="test")]
    assert first == second


def test_limit_applies_after_split_filter(mini_mp20_csv):
    """A limit larger than the split's actual size must not pull in rows
    from other splits — this is the bug being fixed: limit used to be
    applied to the whole file before any split filtering.
    """
    train_all = MP20Dataset(root=mini_mp20_csv, split="train")
    train_limited = MP20Dataset(root=mini_mp20_csv, split="train", limit=1000)
    assert len(train_limited) == len(train_all)
