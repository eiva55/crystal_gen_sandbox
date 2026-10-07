"""match rate / RMSE (StructureMatcher) и сохранение группы для вывода run_bench.py."""
import sys, json
from pathlib import Path
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

out = Path(sys.argv[1])
order = json.load(open(out / 'order.json'))
meta = pd.read_csv('analysis/results/bench_v0_meta.csv').set_index('material_id')
test = pd.read_csv('models/diffcsp_pp/data/mp_20/test.csv', usecols=['material_id', 'cif']).set_index('material_id')
cifs = {int(p.stem): p for p in out.glob('*.cif')}
off = min(cifs)
print(f"CIF: {len(cifs)}, индексы с {off}, целей {len(order)}")
M = {'std': StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10),    # протокол CDVAE/DiffCSP
     'strict': StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)}
rows, refs = [], {}
for i, o in enumerate(order):
    p = cifs.get(i + off)
    row = {**o, **meta.loc[o['material_id']].to_dict(), 'generated': p is not None}
    if p is not None:
        gen = Structure.from_file(str(p))
        mid = o['material_id']
        if mid not in refs:
            refs[mid] = CifParser.from_str(test.loc[mid, 'cif']).parse_structures(primitive=True)[0]
        ref = refs[mid]
        row['aligned'] = gen.composition.reduced_formula == ref.composition.reduced_formula
        for n, m in M.items():
            r = m.get_rms_dist(gen, ref)
            row[f'match_{n}'] = r is not None
            row[f'rmse_{n}'] = None if r is None else r[0]
        for sp in (0.01, 0.1):
            try:
                g = SpacegroupAnalyzer(gen, symprec=sp).get_space_group_number()
            except Exception:
                g = -1
            row[f'sg_ok_{sp}'] = (g == row['sg'])
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(out / 'eval.csv', index=False)
d = df[df.generated & df.aligned.fillna(False).astype(bool)]
print(f"сгенерировано {int(df.generated.sum())}/{len(df)}, выровнено по составу {len(d)}")
print(d.groupby('bin', sort=False)[['match_std', 'match_strict', 'sg_ok_0.01', 'sg_ok_0.1']].mean().round(2))
print(d.groupby('bin', sort=False)[['rmse_std', 'rmse_strict']].mean().round(4))
