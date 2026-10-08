"""Метаданные и ковариаты для всех структур test MP-20 (набор full): для оценки опубликованных генераций CrystaLLM."""
import sys
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
warnings.filterwarnings('ignore')

D = 'models/diffcsp_pp/data/mp_20'
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BINS = [(0, 0.005), (0.005, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.0801)]
g = lambda ds, k: ds[k] if isinstance(ds, dict) else getattr(ds, k)

def anon(f):
    try:
        return Composition(f).anonymized_formula
    except Exception:
        return 'NA'

def n_free(s, ds, symprec=0.1):
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

def work(args):
    mid, cif, sg_tab = args
    warnings.filterwarnings('ignore')
    try:
        s = CifParser.from_str(cif).parse_structures(primitive=True)[0]
        ds = SpacegroupAnalyzer(s, symprec=0.1).get_symmetry_dataset()
        sg = int(g(ds, 'number'))
        return mid, sg, len(s), len(set(g(ds, 'equivalent_atoms'))), n_free(s, ds), int(sg == sg_tab)
    except Exception:
        return mid, -1, 0, 0, -1, 0

if __name__ == '__main__':
    tr = pd.read_csv(f'{D}/train.csv', usecols=['pretty_formula', 'spacegroup.number'])
    tr['proto'] = [anon(f) + '_' + str(s) for f, s in zip(tr.pretty_formula, tr['spacegroup.number'])]
    sg_cnt, proto_cnt = tr['spacegroup.number'].value_counts(), tr.proto.value_counts()
    te = pd.read_csv(f'{D}/test.csv', usecols=['material_id', 'pretty_formula', 'e_above_hull', 'spacegroup.number', 'cif'])
    with Pool(NPROC) as p:
        res = p.map(work, [(m, c, int(s)) for m, c, s in zip(te.material_id, te.cif, te['spacegroup.number'])], chunksize=20)
    r = pd.DataFrame(res, columns=['material_id', 'sg', 'n_atoms', 'n_orbits', 'n_free', 'sg_agree'])
    m = te.drop(columns='cif').rename(columns={'pretty_formula': 'formula'}).merge(r, on='material_id', suffixes=('_tab', ''))
    bad = m[(m.sg < 0) | (m.sg_agree == 0)]
    print(f"целей {len(m)}; не разобрано или группа при symprec 0,1 не совпала с таблицей: {len(bad)} (остаются в наборе, группа = определённая при symprec 0,1)")
    m = m[m.sg > 0].copy()
    m['bin'] = pd.cut(m.e_above_hull, [lo for lo, _ in BINS] + [BINS[-1][1]], right=False, include_lowest=True,
                      labels=[f'{lo}-{hi}' for lo, hi in BINS]).astype(str)
    m['proto'] = [anon(f) + '_' + str(s) for f, s in zip(m.formula, m.sg)]
    m['sg_train_count'] = m.sg.map(sg_cnt).fillna(0).astype(int)
    m['proto_train_count'] = m.proto.map(proto_cnt).fillna(0).astype(int)
    m['split'] = 'test'
    cols = ['material_id', 'formula', 'e_above_hull', 'bin', 'sg', 'n_atoms', 'n_orbits', 'sg_train_count', 'proto_train_count', 'n_free', 'split']
    m[cols].to_csv('analysis/results/bench_full_meta.csv', index=False)
    print(m.groupby('bin', sort=False).agg(N=('material_id', 'size'), n_free0=('n_free', lambda x: int((x == 0).sum())),
                                           med_atoms=('n_atoms', 'median'), med_free=('n_free', 'median')))
