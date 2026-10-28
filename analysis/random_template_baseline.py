"""Опорный вариант без модели: случайные структуры с тем же шаблоном, что получает DiffCSP++.

Для каждой цели набора (configs/csp_targets/bench_<набор>.json: группа, орбиты Вайкоффа, элементы на орбитах) pyxtal
строит K случайных структур с этим шаблоном: свободные координаты орбит берутся случайно. Два варианта решётки:
  random — решётка тоже случайная (pyxtal, плотность по умолчанию): нижняя граница того, что даёт один шаблон;
  oracle — решётка цели (стандартная конвенциональная ячейка spglib): верхняя граница для «шаблон + верная решётка»,
           то есть насколько структура определяется шаблоном и решёткой без выбора координат.
Каждая структура сравнивается с целью (StructureMatcher, строгие 0,2; 0,3; 5° и стандартные 0,3; 0,5; 10°).
Неудача построения (pyxtal не смог расставить атомы) считается неудачей попытки. Кратности орбит в условии DiffCSP++
заданы для примитивной ячейки и пересчитываются в конвенциональную по таблице группы.

Выход: CSV по попыткам и сводка по ролям цели и по числу свободных координат.

Использование (окружение с pyxtal, например diffcsp_pp или mp_lookup):
  python analysis/random_template_baseline.py --set pairs_tv_nf --k 10 --nproc 6 \
    --out analysis/results/random_template_pairs_tv_nf_k10.csv > analysis/results/random_template_pairs_tv_nf_k10.txt
"""
import argparse
import json
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--set', required=True)
ap.add_argument('--k', type=int, default=10)
ap.add_argument('--nproc', type=int, default=4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
ap.add_argument('--data', default='models/diffcsp_pp/data/mp_20')
ap.add_argument('--targets', default='')
a = ap.parse_args()

R = 'analysis/results'
meta = pd.read_csv(f'{R}/bench_{a.set}_meta.csv').set_index('material_id')
tg = json.load(open(a.targets or f'configs/csp_targets/bench_{a.set}.json'))
mp = pd.concat([pd.read_csv(f'{a.data}/{s}.csv', usecols=['material_id', 'cif']) for s in ('train', 'val', 'test')])
CIF = mp[mp.material_id.isin(meta.index)].drop_duplicates('material_id').set_index('material_id').cif.to_dict()


def work(job):
    t, i0 = job
    warnings.filterwarnings('ignore')
    from pymatgen.io.cif import CifParser
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pyxtal import pyxtal
    from pyxtal.lattice import Lattice as PLattice
    from pyxtal.symmetry import Group
    strict = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    std = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    mid = t['label']
    ref = CifParser.from_str(CIF[mid]).parse_structures(primitive=True)[0]
    conv = SpacegroupAnalyzer(ref, symprec=0.1).get_conventional_standard_structure()
    # в условии DiffCSP++ кратности даны для примитивной ячейки; pyxtal работает в конвенциональной: берём букву орбиты
    # и кратность из таблицы группы
    G = Group(t['spacegroup_number'])
    species, sites, nums = [], [], []
    for w, el in zip(t['wyckoff_letters'], t['atom_types']):
        letter = ''.join(ch for ch in w if ch.isalpha())
        wp = G.get_wyckoff_position(letter)
        if el not in species:
            species.append(el); sites.append([]); nums.append(0)
        j = species.index(el)
        sites[j].append(f'{wp.multiplicity}{letter}'); nums[j] += wp.multiplicity
    ltype = G.lattice_type
    L = conv.lattice
    rows = []
    for mode in ('random', 'oracle'):
        for k in range(a.k):
            np.random.seed(a.seed + 1000 * i0 + 100 * (mode == 'oracle') + k)
            row = dict(material_id=mid, mode=mode, k=k, built=False, match_strict=False, match_std=False)
            try:
                c = pyxtal()
                kw = dict(sites=sites)
                if mode == 'oracle':
                    kw['lattice'] = PLattice.from_para(L.a, L.b, L.c, L.alpha, L.beta, L.gamma, ltype=ltype)
                c.from_random(3, t['spacegroup_number'], species, nums, **kw)
                if c.valid:
                    s = c.to_pymatgen()
                    row['built'] = True
                    row['match_strict'] = bool(strict.fit(s, ref))
                    row['match_std'] = bool(std.fit(s, ref))
            except Exception:
                pass
            rows.append(row)
    return rows


jobs = [(t, i) for i, t in enumerate(tg) if t['label'] in CIF]
with Pool(a.nproc) as pool:
    rows = [r for rs in pool.imap_unordered(work, jobs, chunksize=4) for r in rs]
df = pd.DataFrame(rows).merge(meta[['pair_role', 'n_free', 'e_above_hull', 'formula']].reset_index(), on='material_id', how='left')
df.to_csv(a.out, index=False)

print(f'набор {a.set}: целей {df.material_id.nunique()}, попыток на цель и вариант решётки {a.k}')
for mode, g in df.groupby('mode'):
    print(f'\n== решётка: {mode}; построено {g.built.mean():.3f} попыток')
    t = g.groupby('pair_role')[['match_strict', 'match_std']].mean().round(3)
    print('доля совпадений по ролям цели:'); print(t.to_string())
    g = g.assign(nf=pd.cut(g.n_free, [-1, 0, 2, 4, 8, 100], labels=['0', '1–2', '3–4', '5–8', '>8']))
    t = g.groupby('nf', observed=True).agg(целей=('material_id', 'nunique'), строго=('match_strict', 'mean'), станд=('match_std', 'mean')).round(3)
    print('по числу свободных координат:'); print(t.to_string())
    per = g.groupby('material_id').match_strict.max()
    print(f'целей хотя бы с одним строгим совпадением: {int(per.sum())} из {len(per)}')
