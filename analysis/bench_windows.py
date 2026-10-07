"""Окно допуска symprec, в котором структура сохраняет целевую группу, для эталона и сэмпла (прокси амплитуды искажения)."""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

GRID = np.logspace(-3, np.log10(0.5), 25)

def tau(s, tgt):
    """Наибольший допуск сетки, до которого (подряд от малых) структура имеет группу tgt; NaN если не имеет даже при минимальном."""
    last = np.nan
    for sp in GRID:
        try:
            g = SpacegroupAnalyzer(s, symprec=float(sp)).get_space_group_number()
        except Exception:
            g = -1
        if g != tgt:
            break
        last = float(sp)
    return last

test = pd.read_csv('models/diffcsp_pp/data/mp_20/test.csv', usecols=['material_id', 'cif']).set_index('material_id')
rows, ref_cache = [], {}
for ev in sorted(glob.glob('outputs/bench/*/eval.csv')):
    d = pd.read_csv(ev)
    chunk = Path(ev).parent
    cifs = sorted(chunk.glob('*.cif'), key=lambda p: int(p.stem))
    off = int(cifs[0].stem)
    for i, r in d.iterrows():
        mid, tgt = r['material_id'], int(r['sg'])
        if mid not in ref_cache:
            ref = CifParser.from_str(test.loc[mid, 'cif']).parse_structures(primitive=True)[0]
            ref_cache[mid] = tau(ref, tgt)
        f = chunk / f'{off + i}.cif'
        tg = tau(Structure.from_file(str(f)), tgt) if f.exists() else np.nan
        rows.append({'material_id': mid, 'k': r['k'], 'chunk': chunk.name, 'tau_ref': ref_cache[mid], 'tau_gen': tg})
out = pd.DataFrame(rows).drop_duplicates(['material_id', 'k'])
out.to_csv('analysis/results/bench_v0_windows.csv', index=False)
print(len(out), 'структур; tau_gen=NaN:', int(out.tau_gen.isna().sum()), '; tau_ref=NaN:', int(out.tau_ref.isna().sum()))
