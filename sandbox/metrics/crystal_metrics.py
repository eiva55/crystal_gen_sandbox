import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from collections import Counter
from typing import List, Dict, Tuple

class CrystalMetrics:
    @staticmethod
    def compute_validity(structures: List[Structure]) -> float:
        """Доля структур, которые успешно построены и не содержат ошибок."""
        valid = 0
        for s in structures:
            try:
                if s is not None and len(s) > 0:
                    valid += 1
            except:
                pass
        return valid / len(structures) if structures else 0.0

    @staticmethod
    def compute_uniqueness(structures: List[Structure]) -> float:
        """Доля уникальных структур (по сравнению друг с другом)."""
        if len(structures) < 2:
            return 1.0
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
        unique = []
        for s in structures:
            if s is None:
                continue
            is_duplicate = False
            for u in unique:
                if matcher.fit(s, u):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(s)
        return len(unique) / len(structures) if structures else 0.0

    @staticmethod
    def compute_novelty(structures: List[Structure], reference: List[Structure]) -> float:
        """Доля структур, не совпадающих с референсными."""
        if not reference or not structures:
            return 1.0
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
        novel = 0
        for s in structures:
            if s is None:
                continue
            is_novel = True
            for r in reference:
                if matcher.fit(s, r):
                    is_novel = False
                    break
            if is_novel:
                novel += 1
        return novel / len(structures) if structures else 0.0

    @staticmethod
    def compute_all(structures: List[Structure], reference: List[Structure] = None,
                     reference_entries=None) -> Dict[str, float]:
        result = {
            "validity": CrystalMetrics.compute_validity(structures),
            "uniqueness": CrystalMetrics.compute_uniqueness(structures),
            "novelty": CrystalMetrics.compute_novelty(structures, reference) if reference else 1.0,
        }
        if reference_entries is not None:
            # CHGNet-based approximation, NOT the DFT-based S.U.N. reported
            # in papers — see sandbox/metrics/stability.py docstring.
            from sandbox.metrics.stability import compute_stability
            stability = compute_stability(structures, reference_entries)
            result["stable_rate_chgnet_approx"] = stability["stable_rate"]
            result["metastable_rate_chgnet_approx"] = stability["metastable_rate"]
            result["stability_evaluated_count"] = stability["evaluated"]
        return result
