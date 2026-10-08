"""Набор целей из заданной части MP-20 с ковариатами (контроль «train против test» и новые наборы)."""
import argparse
import json
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--set', required=True)
ap.add_argument('--split', default='train', choices=['train', 'test'])
ap.add_argument('--per-bin', type=int, default=60)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()

D = 'models/diffcsp_pp/data/mp_20'
BINS = [(0, 0.005), (0.005, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.0801)]
g = lambda ds, k: ds[k] if isinstance(ds, dict) else getattr(ds, k)

def anon(f):
    try:
        return Composition(f).anonymized_formula
    except Exception:
        return 'NA'

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

tr = pd.read_csv(f'{D}/train.csv', usecols=['pretty_formula', 'spacegroup.number'])
tr['proto'] = [anon(f) + '_' + str(s) for f, s in zip(tr.pretty_formula, tr['spacegroup.number'])]
sg_cnt, proto_cnt = tr['spacegroup.number'].value_counts(), tr.proto.value_counts()
loo = 1 if a.split == 'train' else 0   # для целей из train самого себя из счётчиков вычитаем

src = pd.read_csv(f'{D}/{a.split}.csv', usecols=['material_id', 'pretty_formula', 'e_above_hull', 'spacegroup.number', 'cif'])
targets, meta = [], []
for lo, hi in BINS:
    pool = src[(src.e_above_hull >= lo) & (src.e_above_hull < hi)]
    pool = pool.sample(min(len(pool), a.per_bin * 2), random_state=a.seed)
    kept = 0
    for _, row in pool.iterrows():
        if kept >= a.per_bin:
            break
        try:
            s = CifParser.from_str(row['cif']).parse_structures(primitive=True)[0]
            ds = SpacegroupAnalyzer(s, symprec=0.1).get_symmetry_dataset()
            nf = n_free(s)
        except Exception:
            continue
        if int(g(ds, 'number')) != int(row['spacegroup.number']) or len(s) > 20:
            continue
        species = [site.specie.symbol for site in s]
        groups = defaultdict(list)
        for i, eq in enumerate(g(ds, 'equivalent_atoms')):
            groups[eq].append(i)
        types, letters = [], []
        for idx in groups.values():
            types.append(species[idx[0]])
            letters.append(f"{len(idx)}{g(ds, 'wyckoffs')[idx[0]]}")
        sgn = int(g(ds, 'number'))
        proto = anon(row['pretty_formula']) + '_' + str(sgn)
        targets.append({'label': row['material_id'], 'spacegroup_number': sgn,
                        'wyckoff_letters': letters, 'atom_types': types})
        meta.append({'material_id': row['material_id'], 'formula': row['pretty_formula'],
                     'e_above_hull': row['e_above_hull'], 'bin': f'{lo}-{hi}', 'sg': sgn,
                     'n_atoms': len(s), 'n_orbits': len(groups),
                     'sg_train_count': int(sg_cnt.get(sgn, 0)) - loo,
                     'proto_train_count': int(proto_cnt.get(proto, 0)) - loo,
                     'n_free': nf, 'split': a.split})
        kept += 1
    print(f"корзина {lo}-{hi}: отобрано {kept}")

json.dump(targets, open(f'configs/csp_targets/bench_{a.set}.json', 'w'), indent=1)
m = pd.DataFrame(meta)
m.to_csv(f'analysis/results/bench_{a.set}_meta.csv', index=False)
print(f"набор {a.set} ({a.split}): целей {len(m)}; n_free=0: {(m.n_free == 0).sum()}; прототип без аналогов в train: {(m.proto_train_count <= 0).sum()}")
print(m.groupby('bin', sort=False)[['n_free', 'n_atoms', 'sg_train_count', 'proto_train_count']].median())
