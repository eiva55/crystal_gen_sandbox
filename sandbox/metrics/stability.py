"""Approximate stability metrics via CHGNet single-point energies.

This is NOT the DFT-based S.U.N. reported in the papers we compare against —
it's a cheaper proxy: CHGNet (a pretrained ML potential) predicts energy for
the as-generated structure (no relaxation), and we build a convex hull from
CHGNet energies of a real MP-20 reference subset (not the full Materials
Project database, which we don't have API access to). Treat results as a
rough approximation, not a substitute for real DFT S.U.N. numbers.
"""
import json
import os
from typing import List, Optional

from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry


def _get_calculator():
    from chgnet.model import CHGNetCalculator
    return CHGNetCalculator(use_device="cpu")


def _chgnet_energy(structure: Structure, calculator) -> Optional[float]:
    """Return total (not per-atom) CHGNet-predicted energy, or None on failure."""
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = calculator
        return atoms.get_potential_energy()
    except Exception as exc:
        print(f"CHGNet energy failed for {structure.composition.reduced_formula}: {exc}")
        return None


def build_reference_entries(structures: List[Structure], cache_path: str) -> List[PDEntry]:
    """Build (or load from cache) CHGNet-energy PDEntry list for reference
    structures. Cache is keyed by composition + energy only. Returns a flat
    list of entries — compute_stability() picks the relevant subset per
    structure rather than building one all-elements PhaseDiagram (which is
    computationally intractable at this element count).
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return [PDEntry(composition=e["composition"], energy=e["energy"]) for e in cached]

    calculator = _get_calculator()
    cached = []
    for s in structures:
        energy = _chgnet_energy(s, calculator)
        if energy is not None:
            cached.append({"composition": s.composition.formula, "energy": energy})

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cached, f)

    return [PDEntry(composition=e["composition"], energy=e["energy"]) for e in cached]


def compute_stability(structures: List[Structure], reference_entries: List[PDEntry],
                       stable_threshold: float = 0.0, metastable_threshold: float = 0.1) -> dict:
    """Fraction of structures with CHGNet-predicted e_above_hull below the
    stable/metastable thresholds (eV/atom).

    Builds a small PhaseDiagram per structure, restricted to reference
    entries whose elements are a subset of that structure's elements —
    this is both how e_above_hull is meant to be computed (against the
    relevant chemical subsystem, not the whole periodic table at once) and
    far cheaper: a full 87-element PhaseDiagram is computationally
    intractable (convex hull in ~86 dimensions), while each structure's
    subsystem is typically 2-5 elements.
    """
    calculator = _get_calculator()
    stable, metastable, evaluated = 0, 0, 0

    for s in structures:
        if s is None:
            continue
        energy = _chgnet_energy(s, calculator)
        if energy is None:
            continue

        struct_elements = {el.symbol for el in s.composition.elements}
        relevant_entries = [
            e for e in reference_entries
            if set(e.composition.chemical_system.split("-")) <= struct_elements
        ]
        if not relevant_entries:
            print(f"No reference subsystem for {s.composition.reduced_formula}, skipping")
            continue

        try:
            entry = ComputedEntry(composition=s.composition, energy=energy)
            local_pd = PhaseDiagram(relevant_entries + [entry])
            e_hull = local_pd.get_e_above_hull(entry)
        except Exception as exc:
            print(f"Skipping e_above_hull for {s.composition.reduced_formula}: {exc}")
            continue

        evaluated += 1
        if e_hull < stable_threshold:
            stable += 1
        if e_hull < metastable_threshold:
            metastable += 1

    if evaluated == 0:
        return {"stable_rate": 0.0, "metastable_rate": 0.0, "evaluated": 0}

    return {
        "stable_rate": stable / evaluated,
        "metastable_rate": metastable / evaluated,
        "evaluated": evaluated,
    }
