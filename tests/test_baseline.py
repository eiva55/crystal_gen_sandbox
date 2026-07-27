from sandbox.baselines.random_baseline import RandomMP20CopyBaseline


def test_baseline_returns_zero_novelty_against_its_own_source(mini_mp20_csv):
    from sandbox.metrics.crystal_metrics import CrystalMetrics

    model = RandomMP20CopyBaseline(mp20_root=mini_mp20_csv)
    generated = model.generate(num_samples=2, batch_size=2, device="cpu")

    from sandbox.datasets.mp20 import MP20Dataset
    reference = list(MP20Dataset(root=mini_mp20_csv, split="test"))

    novelty = CrystalMetrics.compute_novelty(generated, reference)
    assert novelty == 0.0
