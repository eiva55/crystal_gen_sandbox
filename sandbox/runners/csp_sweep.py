"""Composition sweep runner for CSP models (DiffCSP++ etc.).

Reads a list of target compositions from a JSON file, calls
model.generate() once per target, saves each result to its own
subdirectory of save_dir so downstream metrics can be computed per
sweep point (e.g. doping fraction -> stability).

Returns a JSON-serializable summary (counts per label), not the raw
Structure objects — run.py unconditionally json.dumps any dict result
into metrics.json, and pymatgen.Structure is not JSON-serializable.
The actual structures are already written to disk as CIFs per label
subdirectory; this return value is just the summary that run.py logs.
"""
import json
import os
from typing import Dict


def run_csp_sweep(model, save_dir: str, targets_path: str,
                   num_samples_per_target: int = 10, batch_size: int = 10,
                   device: str = "cpu", **kwargs) -> Dict[str, int]:
    with open(targets_path) as f:
        targets = json.load(f)

    summary = {}
    for i, target in enumerate(targets):
        label = target.get("label", f"point_{i}")
        point_dir = os.path.join(save_dir, label)
        os.makedirs(point_dir, exist_ok=True)

        structures = model.generate(
            num_samples=num_samples_per_target, batch_size=batch_size,
            device=device, save_dir=point_dir,
            target_composition={k: v for k, v in target.items() if k != "label"},
            **kwargs,
        )
        summary[label] = len(structures)
        print(f"[{label}] generated {len(structures)} structures -> {point_dir}")

    return summary
