import os
import torch
from typing import List, Optional
from pymatgen.core import Structure


def evaluate_generation(model, num_samples: int, batch_size: int, device: torch.device,
                         save_dir: str = None, dataset=None, task=None, viz_enabled: bool = False,
                         metrics=None, stability_reference_path: str = "cache/chgnet_full_reference.json",
                         **kwargs):
    model.to(device)
    structures = model.generate(num_samples, batch_size, device, save_dir=save_dir, **kwargs)
    if not structures:
        print("No structures generated.")
        return {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Prefix is "eval_gen_" (not "gen_") specifically to avoid colliding
        # with whatever filenames a model's own generate() may already have
        # written into this same save_dir — e.g. CrystalDiT's
        # generate_crystals.py defaults to prefix="gen" and writes directly
        # into save_dir, so a plain "gen_{i}.cif" here silently overwrote
        # most of its own output (mismatched 0- vs 1-indexing left one
        # duplicate file behind). This loop is a convenience re-save so CIFs
        # always land in save_dir even for models whose own scripts write
        # elsewhere (ADiT, MiAD, SGEquiDiff) — it's intentionally redundant
        # for models that already write into save_dir themselves, not wrong.
        for i, s in enumerate(structures):
            if s is not None:
                s.to(os.path.join(save_dir, f"eval_gen_{i}.cif"))

    if viz_enabled and save_dir and task is not None:
        task.visualize(structures, save_dir)

    reference_structures = list(dataset) if dataset is not None else None

    if metrics is not None:
        # Config-driven path: metrics is a BaseMetrics-contract object
        # instantiated from cfg.metrics (see configs/metrics/basic.yaml).
        result = metrics.compute(structures, reference_structures)
    else:
        # Backward-compatible fallback for callers that don't pass cfg.metrics.
        from sandbox.metrics.crystal_metrics import CrystalMetrics
        reference_entries = None
        if os.path.exists(stability_reference_path):
            import json
            from sandbox.metrics.stability import PDEntry
            with open(stability_reference_path) as f:
                cached = json.load(f)
            reference_entries = [PDEntry(composition=e["composition"], energy=e["energy"]) for e in cached]
        else:
            print(f"No stability reference found at {stability_reference_path} — skipping stability metrics.")
        result = CrystalMetrics.compute_all(structures, reference_structures, reference_entries=reference_entries)

    print("Metrics:", result)
    return result
