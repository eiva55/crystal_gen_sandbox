import os
import json
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from smact.screening import smact_validity
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from sandbox.contracts import BaseMetrics


class CrystalMetrics(BaseMetrics):
    """Config-instantiable metrics bundle.

    Can still be used purely statically (CrystalMetrics.compute_validity(...),
    .compute_uniqueness(...), .compute_novelty(...), .compute_all(...)) — those
    are unchanged and used directly by the unit tests. The instance method
    `compute()` is the BaseMetrics-contract entrypoint used by the runner when
    metrics are wired through `cfg.metrics` (hydra.utils.instantiate).
    """

    # CDVAE-style structural validity threshold (Å) — used consistently across
    # the CDVAE/DiffCSP/CrystalDiT/ADiT/WyFormer/SGEquiDiff/MiAD lineage per
    # our lit review (matgen_lit_2.xls). Below this, atoms are considered to
    # be unphysically overlapping.
    MIN_INTERATOMIC_DISTANCE = 0.5

    STRUCTURE_MATCHER_KWARGS = dict(ltol=0.2, stol=0.3, angle_tol=5)

    def __init__(self, stability_reference_path: Optional[str] = None,
                 compute_stability: bool = True,
                 relax_fmax: float = 0.1, relax_steps: int = 100):
        self.stability_reference_path = stability_reference_path
        self.compute_stability = compute_stability
        # CHGNet-relaxation controls (Phase 4) — steps is a hard cap, not a
        # guarantee of true convergence; 100 is a pragmatic default given
        # this is already an ML-potential approximation, not DFT, so tight
        # convergence buys less than it costs in wall-clock time at scale.
        self.relax_fmax = relax_fmax
        self.relax_steps = relax_steps

    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------

    @staticmethod
    def compute_structural_validity(structure: Optional[Structure]) -> bool:
        """All pairwise interatomic distances must exceed
        MIN_INTERATOMIC_DISTANCE. Uses pymatgen's periodic (minimum-image)
        distance_matrix, not raw Cartesian distances, so atoms that are close
        across a periodic cell boundary are still caught.
        """
        if structure is None or len(structure) == 0:
            return False
        if len(structure) == 1:
            return True  # nothing to violate distance with
        dm = structure.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        return bool(dm.min() > CrystalMetrics.MIN_INTERATOMIC_DISTANCE)

    @staticmethod
    def compute_compositional_validity(structure: Optional[Structure]) -> bool:
        """SMACT charge-neutrality + Pauling electronegativity check.

        Uses consensus=1, commonality="low" — NOT smact_validity's own
        library defaults (consensus=3, commonality="medium"), which filter
        out oxidation states below a literature-frequency threshold. That
        filtering is a newer addition to the smact library; the
        CDVAE-lineage papers we compare against (CrystalDiT, ADiT,
        SGEquiDiff, MiAD, WyFormer — see matgen_lit_2.xls) predate it and
        effectively used the permissive behavior (any historically observed
        oxidation state counts, no frequency weighting).

        Empirically confirmed on ADiT's N=1000 generated set: library
        defaults gave 64.5% vs. the paper's reported 90.83% (Δ≈-26 п.п.);
        consensus=1/commonality="low" gives 92.9% (Δ≈+2 п.п.) — the
        remaining gap is well within noise/methodology-variance range,
        unlike the default settings' gap. use_pauling_test barely moved
        the number in the same experiment (64.5% -> 64.7%), so the
        commonality/consensus filter was the actual driver, not the
        Pauling electronegativity test.
        """
        if structure is None or len(structure) == 0:
            return False
        try:
            return bool(smact_validity(structure.composition, consensus=1, commonality="low"))
        except Exception as exc:
            print(f"SMACT validity check failed for {structure.composition.reduced_formula}: {exc}")
            return False

    @staticmethod
    def compute_validity(structures: List[Structure]) -> float:
        """Structural AND compositional validity — matches the paper
        convention. Replaces the previous placeholder (non-None and
        non-empty), which always evaluated to ~1.0 regardless of sample
        size or actual structure quality.
        """
        if not structures:
            return 0.0
        valid = 0
        for s in structures:
            if s is None:
                continue
            if (CrystalMetrics.compute_structural_validity(s)
                    and CrystalMetrics.compute_compositional_validity(s)):
                valid += 1
        return valid / len(structures)

    # ------------------------------------------------------------------
    # Shared grouping / filtering helpers
    #
    # Each of these takes the PREVIOUS stage's output and returns the
    # surviving subset (not a rate) — compute_uniqueness/compute_novelty and
    # the gated S.U.N. chain in compute_all both build on the same three
    # filters, so there's exactly one place that defines "duplicate" and
    # "novel", instead of two implementations that could silently diverge.
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_formula(structures: List[Structure]) -> Dict[str, List[Structure]]:
        """Bucket non-None structures by reduced_formula.

        StructureMatcher.fit() already rejects mismatched compositions
        internally, so grouping doesn't change any result — it just avoids
        calling fit() (lattice/site matching setup) on pairs that can never
        match.
        """
        groups: Dict[str, List[Structure]] = defaultdict(list)
        for s in structures:
            if s is None:
                continue
            groups[s.composition.reduced_formula].append(s)
        return groups

    @staticmethod
    def _filter_valid(structures: List[Structure]) -> List[Structure]:
        """Return the subset passing structural AND compositional validity."""
        return [
            s for s in structures
            if s is not None
            and CrystalMetrics.compute_structural_validity(s)
            and CrystalMetrics.compute_compositional_validity(s)
        ]

    @staticmethod
    def _filter_unique(structures: List[Structure]) -> List[Structure]:
        """Return one representative per duplicate-group (the unique subset)."""
        matcher = StructureMatcher(**CrystalMetrics.STRUCTURE_MATCHER_KWARGS)
        unique: List[Structure] = []
        for group in CrystalMetrics._group_by_formula(structures).values():
            kept: List[Structure] = []
            for s in group:
                if not any(matcher.fit(s, u) for u in kept):
                    kept.append(s)
            unique.extend(kept)
        return unique

    @staticmethod
    def _filter_novel(structures: List[Structure], reference: List[Structure]) -> List[Structure]:
        """Return the subset with no match in `reference`."""
        if not reference:
            return list(structures)
        matcher = StructureMatcher(**CrystalMetrics.STRUCTURE_MATCHER_KWARGS)
        reference_by_formula = CrystalMetrics._group_by_formula(reference)
        novel = []
        for s in structures:
            if s is None:
                continue
            candidates = reference_by_formula.get(s.composition.reduced_formula, [])
            if not any(matcher.fit(s, r) for r in candidates):
                novel.append(s)
        return novel

    # ------------------------------------------------------------------
    # Independent (ungated) rates — each computed over the WHOLE generated
    # set on its own axis. Useful diagnostics (and unchanged in meaning from
    # before Phase 3), but NOT what papers report as "S.U.N. rate" — see
    # the gated un_rate / sun_rate_* keys in compute_all for that.
    # ------------------------------------------------------------------

    @staticmethod
    def compute_uniqueness(structures: List[Structure]) -> float:
        """Доля уникальных структур (по сравнению друг с другом)."""
        if len(structures) < 2:
            return 1.0
        unique = CrystalMetrics._filter_unique(structures)
        return len(unique) / len(structures) if structures else 0.0

    @staticmethod
    def compute_novelty(structures: List[Structure], reference: List[Structure]) -> float:
        """Доля структур, не совпадающих с референсными."""
        if not reference or not structures:
            return 1.0
        novel = CrystalMetrics._filter_novel(structures, reference)
        return len(novel) / len(structures) if structures else 0.0

    # ------------------------------------------------------------------
    # Aggregate entrypoints
    # ------------------------------------------------------------------

    @staticmethod
    def compute_all(structures: List[Structure], reference: List[Structure] = None,
                     reference_entries=None,
                     relax_fmax: float = 0.1, relax_steps: int = 100) -> Dict[str, float]:
        n = len(structures) if structures else 0
        structural_valid_count = sum(
            1 for s in structures if s is not None and CrystalMetrics.compute_structural_validity(s)
        ) if structures else 0
        compositional_valid_count = sum(
            1 for s in structures if s is not None and CrystalMetrics.compute_compositional_validity(s)
        ) if structures else 0

        result = {
            # Independent (ungated) — see docstring above the section.
            "structural_validity": structural_valid_count / n if n else 0.0,
            "compositional_validity": compositional_valid_count / n if n else 0.0,
            "validity": CrystalMetrics.compute_validity(structures),
            "uniqueness": CrystalMetrics.compute_uniqueness(structures),
            "novelty": CrystalMetrics.compute_novelty(structures, reference) if reference else 1.0,
        }

        # Gated chain: valid -> unique (among valid) -> novel (among
        # unique-valid, vs reference/train). Matches the paper convention
        # (e.g. CrystalDiT: "63.28% уникальных и новых структур"), где
        # each stage filters the PREVIOUS stage's survivors rather than the
        # whole generated set independently.
        non_none = [s for s in structures if s is not None] if structures else []
        valid = CrystalMetrics._filter_valid(non_none)
        unique_valid = CrystalMetrics._filter_unique(valid)
        un_set = CrystalMetrics._filter_novel(unique_valid, reference) if reference else unique_valid
        result["un_rate"] = len(un_set) / n if n else 0.0

        if reference_entries is not None:
            from sandbox.metrics.stability import compute_stability

            # Ungated (unchanged in meaning from before Phase 3): CHGNet
            # stability over the WHOLE generated set, independent of
            # validity/uniqueness/novelty. Now uses relaxed energies
            # (Phase 4) instead of single-point.
            stability, energy_cache = compute_stability(
                structures, reference_entries, fmax=relax_fmax, steps=relax_steps
            )
            result["stable_rate_chgnet_approx"] = stability["stable_rate"]
            result["metastable_rate_chgnet_approx"] = stability["metastable_rate"]
            result["stability_evaluated_count"] = stability["evaluated"]

            # Gated: stability computed only among the UN (unique+novel+
            # valid) subset — combines with un_rate into the paper-style
            # joint S.U.N./M.S.U.N. rate. `energy_cache` is reused from the
            # call above — every structure in un_set is the SAME object
            # (by identity) as in `structures` (filters don't copy), so this
            # call hits the cache for all of them instead of re-relaxing.
            if un_set:
                stability_un, energy_cache = compute_stability(
                    un_set, reference_entries, fmax=relax_fmax, steps=relax_steps,
                    energy_cache=energy_cache,
                )
                result["stable_among_un_rate"] = stability_un["stable_rate"]
                result["metastable_among_un_rate"] = stability_un["metastable_rate"]
                result["un_stability_evaluated_count"] = stability_un["evaluated"]
            else:
                result["stable_among_un_rate"] = 0.0
                result["metastable_among_un_rate"] = 0.0
                result["un_stability_evaluated_count"] = 0

            result["sun_rate_chgnet_approx"] = result["un_rate"] * result["stable_among_un_rate"]
            result["msun_rate_chgnet_approx"] = result["un_rate"] * result["metastable_among_un_rate"]

        return result

    def _load_reference_entries(self):
        if not self.compute_stability or not self.stability_reference_path:
            return None
        from sandbox.metrics.stability import load_cached_reference_entries
        entries = load_cached_reference_entries(self.stability_reference_path)
        if entries is None:
            print(f"No usable stability reference at {self.stability_reference_path} — "
                  "skipping stability metrics.")
        return entries

    def compute(self, generated: List[Structure], reference: Optional[List[Structure]] = None) -> Dict[str, float]:
        """BaseMetrics contract entrypoint — used by the runner via cfg.metrics."""
        reference_entries = self._load_reference_entries()
        return CrystalMetrics.compute_all(
            generated, reference, reference_entries=reference_entries,
            relax_fmax=self.relax_fmax, relax_steps=self.relax_steps,
        )
