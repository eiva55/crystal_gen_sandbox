"""Approximate stability metrics via CHGNet.

Historically this used a single-point (no relaxation) CHGNet energy on the
as-generated structure. That's now upgraded to a CHGNet-driven geometry
relaxation (ASE FIRE optimizer via chgnet.model.StructOptimizer) before
computing the energy — much closer to what the literature does (CHGNet/
M3GNet/ORB relaxation, often followed by DFT on top; see matgen_lit_2.xls).
This is STILL NOT the DFT-based S.U.N. reported in the papers — it's a
cheaper ML-potential-relaxed proxy, not a substitute for real DFT relaxation
+ energy. Treat results as a much closer approximation than the old
single-point version, not as DFT-equivalent.

The reference convex hull is built from a real MP-20 subset (not the full
Materials Project database, which we don't have API access to), with
entries ALSO relaxed the same way — comparing a relaxed generated
structure's energy against an unrelaxed reference hull would be comparing
two different things.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry

# Bumped whenever the methodology behind cached reference energies changes,
# so a cache built under the old single-point (unrelaxed) methodology is
# never silently reused as if it were relaxed — see load_cached_reference_entries.
REFERENCE_CACHE_VERSION = "chgnet_relaxed_v1"


def _get_optimizer():
    from chgnet.model import StructOptimizer
    return StructOptimizer(use_device="cpu")


def _chgnet_relaxed_energy(structure: Structure, optimizer, fmax: float = 0.1,
                            steps: int = 100) -> Optional[float]:
    """Relax `structure` with CHGNet+ASE (FIRE, FrechetCellFilter, cell
    relaxation on) and return the TOTAL (not per-atom) energy at the final
    recorded step, or None on failure. `steps` is a hard cap, not a
    guarantee of convergence to `fmax` — a structure that hasn't converged
    by `steps` still returns its last-step energy rather than raising,
    since a partially-relaxed estimate is still far closer to the true
    minimum than the unrelaxed single-point energy this replaces.
    """
    try:
        result = optimizer.relax(structure, fmax=fmax, steps=steps, verbose=False)
        energies = result["trajectory"].energies
        if not energies:
            return None
        return float(energies[-1])
    except Exception as exc:
        print(f"CHGNet relaxation failed for {structure.composition.reduced_formula}: {exc}")
        return None


def load_cached_reference_entries(cache_path: str) -> Optional[List[PDEntry]]:
    """Load reference PDEntry list from cache_path, but ONLY if it was built
    under the current REFERENCE_CACHE_VERSION. A cache from before the
    relaxation change (a bare JSON list, no version tag) is refused rather
    than silently trusted — comparing a relaxed generated structure's
    energy against an unrelaxed reference hull would give meaningless
    e_above_hull values.
    """
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        cached = json.load(f)
    if not (isinstance(cached, dict) and cached.get("version") == REFERENCE_CACHE_VERSION):
        print(f"Stability reference cache at {cache_path} is stale or from an "
              f"older methodology (expected version={REFERENCE_CACHE_VERSION}) — "
              "refusing to use it. Rebuild with build_reference_entries(...).")
        return None
    entries = cached["entries"]
    return [PDEntry(composition=e["composition"], energy=e["energy"]) for e in entries]


def build_reference_entries(structures: List[Structure], cache_path: str,
                             fmax: float = 0.1, steps: int = 100) -> List[PDEntry]:
    """Build (or load from cache, if current-version) CHGNet-relaxed-energy
    PDEntry list for reference structures. Returns a flat list of entries —
    compute_stability() picks the relevant subset per structure rather than
    building one all-elements PhaseDiagram (computationally intractable at
    this element count).
    """
    cached_entries = load_cached_reference_entries(cache_path)
    if cached_entries is not None:
        return cached_entries

    optimizer = _get_optimizer()
    cached = []
    for s in structures:
        energy = _chgnet_relaxed_energy(s, optimizer, fmax=fmax, steps=steps)
        if energy is not None:
            cached.append({"composition": s.composition.formula, "energy": energy})

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"version": REFERENCE_CACHE_VERSION, "entries": cached}, f)

    return [PDEntry(composition=e["composition"], energy=e["energy"]) for e in cached]


def compute_stability(structures: List[Structure], reference_entries: List[PDEntry],
                       stable_threshold: float = 0.0, metastable_threshold: float = 0.1,
                       fmax: float = 0.1, steps: int = 100,
                       energy_cache: Optional[Dict[int, Optional[float]]] = None
                       ) -> Tuple[dict, Dict[int, Optional[float]]]:
    """Fraction of structures with CHGNet-relaxed-energy e_above_hull below
    the stable/metastable thresholds (eV/atom).

    Builds a small PhaseDiagram per structure, restricted to reference
    entries whose elements are a subset of that structure's elements — both
    the physically correct way to compute e_above_hull (against the
    relevant chemical subsystem, not the whole periodic table) and far
    cheaper than one full-element PhaseDiagram.

    `energy_cache`, if given, is a dict keyed by id(structure) that this
    function reads from and writes to in place — pass the SAME dict across
    multiple calls on overlapping structure lists (e.g. the full generated
    set, then its UN-gated subset) to avoid relaxing the same structure
    object twice. Returns (result_dict, energy_cache) so the caller can keep
    reusing the cache across calls.
    """
    optimizer = _get_optimizer()
    if energy_cache is None:
        energy_cache = {}
    stable, metastable, evaluated = 0, 0, 0

    for s in structures:
        if s is None:
            continue
        key = id(s)
        if key in energy_cache:
            energy = energy_cache[key]
        else:
            energy = _chgnet_relaxed_energy(s, optimizer, fmax=fmax, steps=steps)
            energy_cache[key] = energy
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
        return {"stable_rate": 0.0, "metastable_rate": 0.0, "evaluated": 0}, energy_cache

    return {
        "stable_rate": stable / evaluated,
        "metastable_rate": metastable / evaluated,
        "evaluated": evaluated,
    }, energy_cache
