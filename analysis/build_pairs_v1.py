"""Набор «стабильный против метастабильного полиморфа одного состава» (pairs_v1) и набор отдельных фаз (special_v1).

pairs_v1: составы тестовой части MP-20, у которых есть и почти стабильный (E < 5 мэВ/атом), и метастабильный
(E >= 5 мэВ/атом) полиморф. Берётся самый низкий по энергии почти стабильный полиморф и все метастабильные.
special_v1: явно заданные material_id из любой части MP-20 (например, фазы HfO2 и ZrO2).

Пишет:
  analysis/results/bench_<set>_meta.csv        — все цели (для CrystaLLM)
  analysis/results/bench_<set>nf_meta.csv      — только цели с n_free >= 1 (для DiffCSP++: при n_free = 0 шаблон задаёт структуру целиком)
  configs/csp_targets/bench_<set>.json, bench_<set>nf.json — условия DiffCSP++ (группа, орбиты Вайкоффа, виды атомов)

Пары можно брать из другой части MP-20 (--split val) и исключать составы, уже вошедшие в другой набор
(--exclude-sets pairs_v1): так получаются новые составы для того же сравнения.

Использование:
  python analysis/build_pairs_v1.py                                  # pairs_v1
  python analysis/build_pairs_v1.py --set pairs_val_v1 --split val --exclude-sets pairs_v1
  python analysis/build_pairs_v1.py --set special_v1 --ids mp-352,mp-685097
"""
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
ap.add_argument('--set', default='pairs_v1')
ap.add_argument('--ids', default='', help='список material_id через запятую (режим отдельных фаз)')
ap.add_argument('--stable-max', type=float, default=0.005, help='граница «почти стабильный», эВ/атом')
ap.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='из какой части MP-20 брать пары')
ap.add_argument('--exclude-sets', nargs='*', default=[], help='исключить составы, вошедшие в эти наборы')
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


def bin_of(e):
    for lo, hi in BINS:
        if lo <= e < hi:
            return f'{lo}-{hi}'
    return 'вне диапазона'


cols = ['material_id', 'pretty_formula', 'e_above_hull', 'spacegroup.number', 'cif']
allmp = pd.concat([pd.read_csv(f'{D}/{s}.csv', usecols=cols).assign(split=s) for s in ('train', 'val', 'test')])
tr = allmp[allmp.split == 'train']
tr_proto = (tr.pretty_formula.map(anon) + '_' + tr['spacegroup.number'].astype(str)).value_counts()
sg_cnt = tr['spacegroup.number'].value_counts()
gs = allmp.groupby('pretty_formula').e_above_hull.min()           # самый низкий полиморф состава во всём MP-20
gs_split = allmp.sort_values('e_above_hull').drop_duplicates('pretty_formula').set_index('pretty_formula').split

if a.ids:
    ids = [x.strip() for x in a.ids.split(',') if x.strip()]
    sel = allmp[allmp.material_id.isin(ids)].copy()
    missing = set(ids) - set(sel.material_id)
    if missing:
        print('нет в MP-20:', ', '.join(sorted(missing)))
    sel['pair_role'] = np.where(sel.e_above_hull < a.stable_max, 'stable', 'meta')
else:
    te = allmp[allmp.split == a.split].copy()
    excl = set()
    for es in a.exclude_sets:
        excl |= set(pd.read_csv(f'analysis/results/bench_{es}_meta.csv').formula)
    if excl:
        n0 = te.pretty_formula.nunique()
        te = te[~te.pretty_formula.isin(excl)]
        print(f'исключено составов из {", ".join(a.exclude_sets)}: {n0 - te.pretty_formula.nunique()}')
    te['st'] = te.e_above_hull < a.stable_max
    parts = []
    for f, grp in te.groupby('pretty_formula'):
        if grp.st.any() and (~grp.st).any():
            parts.append(grp[grp.st].nsmallest(1, 'e_above_hull').assign(pair_role='stable'))
            parts.append(grp[~grp.st].assign(pair_role='meta'))
    sel = pd.concat(parts)

targets, meta, skipped = [], [], []
for _, row in sel.sort_values(['pretty_formula', 'e_above_hull']).iterrows():
    try:
        s = CifParser.from_str(row['cif']).parse_structures(primitive=True)[0]
        ds = SpacegroupAnalyzer(s, symprec=0.1).get_symmetry_dataset()
        nf = n_free(s)
    except Exception as e:
        skipped.append((row.material_id, f'разбор: {e}'))
        continue
    sgn = int(g(ds, 'number'))
    if sgn != int(row['spacegroup.number']):
        skipped.append((row.material_id, f'группа {sgn} при допуске 0,1 Å вместо {row["spacegroup.number"]}'))
        continue
    species = [site.specie.symbol for site in s]
    groups = defaultdict(list)
    for i, eq in enumerate(g(ds, 'equivalent_atoms')):
        groups[eq].append(i)
    types, letters = [], []
    for idx in groups.values():
        types.append(species[idx[0]])
        letters.append(f"{len(idx)}{g(ds, 'wyckoffs')[idx[0]]}")
    loo = 1 if row.split == 'train' else 0          # цель из train вычитается из собственных счётчиков
    proto = anon(row.pretty_formula) + '_' + str(sgn)
    targets.append({'label': row.material_id, 'spacegroup_number': sgn, 'wyckoff_letters': letters, 'atom_types': types})
    meta.append({'material_id': row.material_id, 'formula': row.pretty_formula, 'e_above_hull': row.e_above_hull,
                 'bin': bin_of(row.e_above_hull), 'sg': sgn, 'n_atoms': len(s), 'n_orbits': len(groups),
                 'sg_train_count': int(sg_cnt.get(sgn, 0)) - loo, 'proto_train_count': int(tr_proto.get(proto, 0)) - loo,
                 'n_free': nf, 'split': row.split, 'pair_role': row.pair_role,
                 'formula_min_E_mp20': gs[row.pretty_formula], 'formula_min_E_split': gs_split[row.pretty_formula]})

m = pd.DataFrame(meta)
nf_ids = set(m.material_id[m.n_free >= 1])
m.to_csv(f'analysis/results/bench_{a.set}_meta.csv', index=False)
m[m.material_id.isin(nf_ids)].to_csv(f'analysis/results/bench_{a.set}nf_meta.csv', index=False)
json.dump(targets, open(f'configs/csp_targets/bench_{a.set}.json', 'w'), indent=1)
json.dump([t for t in targets if t['label'] in nf_ids], open(f'configs/csp_targets/bench_{a.set}nf.json', 'w'), indent=1)

print(f"набор {a.set}: составов {m.formula.nunique()}, целей {len(m)} (стабильных {(m.pair_role == 'stable').sum()}, "
      f"метастабильных {(m.pair_role == 'meta').sum()}); n_free >= 1: {len(nf_ids)}; пропущено {len(skipped)}")
for mid, why in skipped[:10]:
    print(f'  пропущено {mid}: {why}')
print(m.groupby('bin').size().to_string())
if not a.ids:
    lower = m[m.pair_role == 'stable']
    lower = lower[lower.formula_min_E_split != a.split]
    print(f"составов, у которых ещё ниже по энергии полиморф лежит в другой части MP-20: {lower.formula.nunique()}")
