"""Ближайшие структуры в обучающей выборке для целей бенчмарка: есть ли почти дубликат того же типа."""
import glob
import sys
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings('ignore')

D = 'models/diffcsp_pp/data/mp_20'
CAP = 400         # максимум кандидатов на прототип (случайная подвыборка, seed 0); при превышении число соседей занижено (флаг capped)
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 6

def anon(f):
    try:
        return Composition(f).anonymized_formula
    except Exception:
        return 'NA'

def parse(cif):
    try:
        return CifParser.from_str(cif).parse_structures(primitive=True)[0]
    except Exception:
        return None

# цели: все наборы бенчмарка (в наборе v0 колонки split нет, это test)
metas = []
for p in sorted(glob.glob('analysis/results/bench_*_meta.csv')):
    m = pd.read_csv(p)
    if 'split' not in m:
        m['split'] = 'test'
    metas.append(m[['material_id', 'formula', 'sg', 'split']])
tg = pd.concat(metas).drop_duplicates('material_id').reset_index(drop=True)
tg['proto'] = [anon(f) + '_' + str(s) for f, s in zip(tg.formula, tg.sg)]
print(f"целей: {len(tg)} (test {(tg.split == 'test').sum()}, train {(tg.split == 'train').sum()}), прототипов {tg.proto.nunique()}")

tr = pd.read_csv(f'{D}/train.csv', usecols=['material_id', 'pretty_formula', 'spacegroup.number', 'cif'])
tr['proto'] = [anon(f) + '_' + str(s) for f, s in zip(tr.pretty_formula, tr['spacegroup.number'])]
cif_of = dict(zip(tr.material_id, tr.cif))
if (tg.split == 'test').any():
    te = pd.read_csv(f'{D}/test.csv', usecols=['material_id', 'cif'])
    cif_of.update(dict(zip(te.material_id, te.cif)))

need = set(tg.proto)
cand = tr[tr.proto.isin(need)]
cand = cand.sample(frac=1, random_state=0)
cand = cand[cand.groupby('proto').cumcount() < CAP]
print(f"кандидатов из обучающей выборки: {len(cand)} (до ограничения: {int(tr.proto.isin(need).sum())})")

if __name__ == '__main__':
    with Pool(NPROC) as pool:
        cs = pool.map(parse, list(cand.cif), chunksize=50)
        rs = pool.map(parse, [cif_of[m] for m in tg.material_id], chunksize=20)
    CANDS = {}
    for pr, cid, s in zip(cand.proto, cand.material_id, cs):
        if s is not None:
            CANDS.setdefault(pr, []).append((cid, s.composition.reduced_formula, s))
    REFS = dict(zip(tg.material_id, rs))
    total_cand = tr.proto.value_counts()
    M = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    def work(i):
        r = tg.iloc[i]
        ref = REFS[r.material_id]
        out = {'material_id': r.material_id, 'n_cand': 0, 'n_same_chem': 0, 'n_same_type': 0, 'nn_rms': np.nan, 'capped': False}
        if ref is None:
            return out
        rf = ref.composition.reduced_formula
        out['capped'] = bool(total_cand.get(r.proto, 0) > CAP)
        for cid, cf, s in CANDS.get(r.proto, []):
            if cid == r.material_id:
                continue
            out['n_cand'] += 1
            try:
                rms = M.get_rms_anonymous(ref, s)[0]
                if rms is not None:
                    out['n_same_type'] += 1
                    out['nn_rms'] = rms if np.isnan(out['nn_rms']) else min(out['nn_rms'], rms)
                if cf == rf and M.fit(ref, s):
                    out['n_same_chem'] += 1
            except Exception:
                pass
        return out

    # fork: глобальные словари наследуются процессами без копирования
    with Pool(NPROC) as pool:
        res = pool.map(work, range(len(tg)), chunksize=5)
    nb = pd.DataFrame(res).merge(tg[['material_id', 'split', 'proto']], on='material_id')
    nb.to_csv('analysis/results/bench_neighbors.csv', index=False)
    print("\nдоли целей с хотя бы одним соседом в обучающей выборке:")
    print(nb.groupby('split').agg(N=('material_id', 'size'), same_type=('n_same_type', lambda x: (x > 0).mean()),
                                  same_chem=('n_same_chem', lambda x: (x > 0).mean()),
                                  median_n_same_type=('n_same_type', 'median'),
                                  median_nn_rms=('nn_rms', 'median'), capped=('capped', 'mean')).round(3))
