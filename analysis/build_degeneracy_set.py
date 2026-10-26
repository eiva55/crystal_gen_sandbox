"""Набор целей для гипотезы 4: шаблоны, общие для двух полиморфов одного состава (из template_degeneracy.py).

Для каждой пары (нижний по энергии, верхний) целью берётся нижний полиморф; его шаблон совпадает с шаблоном верхнего
(проверяется). Генерация — run_pairs_diffcsp.sh <набор> <N> 1 (одна цель на часть, N образцов); разбор —
cross_match.py: «цель» — образец совпал с нижним полиморфом, «другой полиморф» с matched_id верхнего — с верхним.

Пишет analysis/results/bench_<name>_meta.csv и configs/csp_targets/bench_<name>.json.
Использование (окружение mp_lookup):
  python analysis/build_degeneracy_set.py --name deg_v1 --pairs mp-1025450:mp-1025523 mp-1094177:mp-1094190 mp-559676:mp-1379
"""
import argparse
import json
from collections import Counter, defaultdict
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--pairs', nargs='+', required=True, help='нижний:верхний')
ap.add_argument('--data', default='models/diffcsp_pp/data/mp_20')
a = ap.parse_args()

cols = ['material_id', 'pretty_formula', 'e_above_hull', 'spacegroup.number', 'cif']
mp = pd.concat([pd.read_csv(f'{a.data}/{s}.csv', usecols=cols).assign(split=s) for s in ('train', 'val', 'test')])
mp = mp.set_index('material_id')
deg = pd.read_csv('analysis/results/template_degeneracy.csv')


def g(ds, k):
    return ds[k] if isinstance(ds, dict) else getattr(ds, k)


def entry(mid):
    s = CifParser.from_str(mp.cif[mid]).parse_structures(primitive=True)[0]
    ds = SpacegroupAnalyzer(s, symprec=0.1).get_symmetry_dataset()
    species = [site.specie.symbol for site in s]
    orbits = defaultdict(list)
    for i, eq in enumerate(g(ds, 'equivalent_atoms')):
        orbits[eq].append(i)
    types = [species[idx[0]] for idx in orbits.values()]
    letters = [f"{len(idx)}{g(ds, 'wyckoffs')[idx[0]]}" for idx in orbits.values()]
    return {'label': mid, 'spacegroup_number': int(g(ds, 'number')), 'wyckoff_letters': letters, 'atom_types': types}, len(s)


def bin_of(e):
    for lo, hi in ((0, 0.005), (0.005, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08)):
        if lo <= e < hi:
            return f'{lo:g}-{hi:g}'
    return '0.06-0.08'


targets, meta = [], []
for p in a.pairs:
    lo, hi = p.split(':')
    t_lo, n_lo = entry(lo)
    t_hi, _ = entry(hi)
    same = (t_lo['spacegroup_number'] == t_hi['spacegroup_number'] and
            Counter(zip(t_lo['atom_types'], t_lo['wyckoff_letters'])) == Counter(zip(t_hi['atom_types'], t_hi['wyckoff_letters'])))
    if not same:
        raise SystemExit(f'{lo} и {hi}: шаблоны различаются')
    if mp.pretty_formula[lo] != mp.pretty_formula[hi] or mp.e_above_hull[lo] >= mp.e_above_hull[hi]:
        raise SystemExit(f'{lo}:{hi}: нужен один состав и нижний по энергии первым')
    d = deg[deg.material_id.isin([lo, hi])].set_index('material_id')
    targets.append(t_lo)
    meta.append({'material_id': lo, 'formula': mp.pretty_formula[lo], 'e_above_hull': mp.e_above_hull[lo],
                 'bin': bin_of(mp.e_above_hull[lo]), 'sg': t_lo['spacegroup_number'], 'n_atoms': n_lo,
                 'split': mp.split[lo], 'pair_role': 'stable', 'alt_id': hi, 'alt_e_above_hull': mp.e_above_hull[hi],
                 'alt_split': mp.split[hi],
                 'train_lo': int(d.train_this_arrangement.get(lo, -1)), 'train_hi': int(d.train_this_arrangement.get(hi, -1)),
                 'train_same_anon': int(d.train_same_anon.get(lo, -1))})
m = pd.DataFrame(meta)
m.to_csv(f'analysis/results/bench_{a.name}_meta.csv', index=False)
json.dump(targets, open(f'configs/csp_targets/bench_{a.name}.json', 'w'), indent=1)
print(m[['material_id', 'formula', 'sg', 'n_atoms', 'e_above_hull', 'split', 'alt_id', 'alt_e_above_hull', 'alt_split',
         'train_lo', 'train_hi', 'train_same_anon']].to_string(index=False))
