"""Число свободных координатных параметров орбит Вайкоффа (контроль сложности шаблона)."""
import numpy as np, pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

meta = pd.read_csv('analysis/results/bench_v0_meta.csv')
test = pd.read_csv('models/diffcsp_pp/data/mp_20/test.csv', usecols=['material_id', 'cif']).set_index('material_id')
g = lambda ds, k: ds[k] if isinstance(ds, dict) else getattr(ds, k)

def n_free(s, symprec=0.1):
    ds = SpacegroupAnalyzer(s, symprec=symprec).get_symmetry_dataset()
    rots, trans, eq = g(ds, 'rotations'), g(ds, 'translations'), g(ds, 'equivalent_atoms')
    pos, total = s.frac_coords, 0
    for rep in sorted(set(eq)):
        x, rows = pos[rep], []
        for R, t in zip(rots, trans):
            d = (R @ x + t) - x
            if np.allclose(d - np.round(d), 0, atol=symprec / 5):
                rows.append(R - np.eye(3))
        total += 3 - (np.linalg.matrix_rank(np.vstack(rows), tol=1e-6) if rows else 0)
    return int(total)

out = []
for mid in meta.material_id:
    s = CifParser.from_str(test.loc[mid, 'cif']).parse_structures(primitive=True)[0]
    out.append({'material_id': mid, 'n_free': n_free(s)})
pd.DataFrame(out).to_csv('analysis/results/bench_v0_covariates.csv', index=False)
d = meta.merge(pd.DataFrame(out), on='material_id')
print(d.groupby('bin', sort=False)[['n_free', 'n_orbits', 'n_atoms']].median())
print(d[['n_free', 'n_orbits', 'n_atoms', 'sg_train_count', 'e_above_hull']].corr().round(2))
