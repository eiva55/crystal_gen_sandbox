import os
from collections import Counter
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Structure

from sandbox.contracts.base import BaseTask


class CrystalGenerationTask(BaseTask):
    def run(self, model, num_samples, batch_size, device, save_dir=None, **kwargs):
        print(f"Running generation task with model {model.__class__.__name__}")
        structures = model.generate(num_samples, batch_size, device, save_dir=save_dir, **kwargs)
        print(f"Generated {len(structures)} structures.")
        return structures

    @staticmethod
    def visualize(structures: List[Structure], outdir: str) -> Optional[str]:
        """Plot aggregated element distribution across generated structures.

        Mirrors the reference repo's task.visualize() pattern (a single
        static plot saved to outdir) — the crystal-generation analogue of a
        forecast plot is an element-composition histogram, since there's no
        time axis to plot here.
        """
        valid = [s for s in structures if s is not None]
        if not valid:
            print("No valid structures to visualize.")
            return None

        element_counts = Counter()
        for s in valid:
            for site in s:
                element_counts[site.specie.symbol] += 1

        elements = sorted(element_counts, key=element_counts.get, reverse=True)
        counts = [element_counts[e] for e in elements]

        plt.figure(figsize=(10, 5))
        plt.bar(elements, counts)
        plt.xlabel("Element")
        plt.ylabel("Atom count")
        plt.title(f"Element distribution across {len(valid)} generated structures")
        plt.tight_layout()

        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "generation_visualization.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved visualization to {path}")
        return path
