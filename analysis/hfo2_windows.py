"""Окно допусков symprec, где структура имеет целевую группу (прокси амплитуды полярного искажения)."""
import sys, json
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

run_dir = Path(sys.argv[1])
GRID = np.logspace(-4, np.log10(0.5), 40)

def sg(s, sp):
    try:
        return SpacegroupAnalyzer(s, symprec=float(sp)).get_space_group_number()
    except Exception:
        return -1

def window(s, tgt):
    ok = [sg(s, sp) == tgt for sp in GRID]
    if not any(ok):
        return None
    i0 = ok.index(True)
    hi = next((float(GRID[j]) for j in range(i0, len(GRID)) if not ok[j]), float('inf'))
    return float(GRID[i0]), hi

def cell(s):
    return sorted(round(x, 2) for x in s.lattice.abc), round(s.volume / len(s), 2)

load = lambda p: Structure.from_dict(json.load(open(p)))
for name, f, tgt in [('эталон mp-352 (SG14)', 'mp-352', 14), ('эталон mp-685097 (SG29)', 'mp-685097', 29)]:
    s = load(f'analysis/refs/{f}.json')
    print(f"{name}: окно {window(s, tgt)}, abc/объём на атом {cell(s)}")

res = {}
for label, tgt in [('hfo2_stable', 14), ('hfo2_meta', 29)]:
    rows = []
    for p in sorted((run_dir / label).glob('*.cif'), key=lambda p: (len(p.stem), p.stem)):
        s = Structure.from_file(str(p))
        w = window(s, tgt)
        rows.append((p.name, None if w is None else w[1], cell(s)))
    his = np.array([np.nan if r[1] is None else r[1] for r in rows])
    fin = his[np.isfinite(his)]
    print(f"\n{label}: N={len(rows)}, держат группу до 0.5 Å: {int(np.isposinf(his).sum())}, теряют раньше: {len(fin)}")
    if len(fin):
        print("  верхняя граница окна, Å (min/10%/25%/50%/75%/max):",
              np.round(np.percentile(fin, [0, 10, 25, 50, 75, 100]), 3))
        for n, hi, c in rows:
            if hi is not None and np.isfinite(hi) and hi < 0.25:
                print(f"    {n:<8} hi={hi:.3f}  abc={c[0]}  V/атом={c[1]}")
    res[label] = rows
Path('analysis/results').mkdir(parents=True, exist_ok=True)
json.dump(res, open('analysis/results/hfo2_windows.json', 'w'), indent=1, default=str)
