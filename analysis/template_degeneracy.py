"""Шаблоны, которые не различают полиморфы: кандидаты для гипотезы 4 (энергия раскладки против частоты в обучении).

Шаблон структуры — то, что получает DiffCSP++: группа, орбиты Вайкоффа примитивной ячейки и элемент на каждой
орбите (те же правила, что в build_pairs_v1.py, допуск симметрии 0,1 Å). Анонимный шаблон — тот же шаблон с точностью
до переименования элементов: группа, анонимная формула и отсортированный набор «буквы орбит одного элемента».

Этапы:
  1. шаблоны всех структур MP-20 (train, val, test) -> analysis/results/mp20_templates.csv (кэш, повторно не считается);
  2. составы, у которых хотя бы два полиморфа с одинаковым шаблоном не совпадают по StructureMatcher (0,2; 0,3; 5°):
     условие на шаблон эти полиморфы не различает;
  3. для каждой такой группы — структуры обучающей части с тем же анонимным шаблоном (другие составы, сама группа
     исключена), отнесённые к одному из полиморфов сравнением с заменой элементов (StructureMatcher.fit_anonymous):
     частота каждой раскладки в обучении.

Выход: analysis/results/template_degeneracy.csv (строка — полиморф в группе) и сводка в stdout. Приоритет для проверки:
более частая в обучении раскладка выше по энергии (частота и энергия указывают в разные стороны) либо частоты равны.

Ограничение: ищутся только полиморфы, записанные в базе с одним шаблоном. Случай шпинели MgAl2O4 (упорядоченная
раскладка записана в Fd-3m, а шаблон R3m её допускает) так не находится: для этого нужно описывать структуру
в подгруппах, это не сделано.

Использование (окружение mp_lookup):
  python analysis/template_degeneracy.py --nproc 4 > analysis/results/template_degeneracy.txt
"""
import argparse
import os
import warnings
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='models/diffcsp_pp/data/mp_20')
ap.add_argument('--nproc', type=int, default=4)
ap.add_argument('--max-train', type=int, default=300, help='не больше стольких структур обучающей части на анонимный шаблон')
a = ap.parse_args()
R = 'analysis/results'
CACHE = f'{R}/mp20_templates.csv'


def g(ds, k):
    return ds[k] if isinstance(ds, dict) else getattr(ds, k)


def template(item):
    mid, cif = item
    warnings.filterwarnings('ignore')
    try:
        s = CifParser.from_str(cif).parse_structures(primitive=True)[0]
        ds = SpacegroupAnalyzer(s, symprec=0.1).get_symmetry_dataset()
    except Exception:
        return mid, None, None, None, None
    species = [site.specie.symbol for site in s]
    orbits = defaultdict(list)
    for i, eq in enumerate(g(ds, 'equivalent_atoms')):
        orbits[eq].append(i)
    per_el = defaultdict(list)
    for idx in orbits.values():
        per_el[species[idx[0]]].append(f"{len(idx)}{g(ds, 'wyckoffs')[idx[0]]}")
    sgn = int(g(ds, 'number'))
    tpl = f"{sgn}|" + ';'.join(f"{el}:{','.join(sorted(v))}" for el, v in sorted(per_el.items()))
    anon_parts = sorted(','.join(sorted(v)) for v in per_el.values())
    anon_f = ','.join(f'{x:g}' for x in sorted(Composition(s.composition).reduced_composition.values()))
    return mid, sgn, tpl, f"{sgn}|{anon_f}|" + ';'.join(anon_parts), len(s)


cols = ['material_id', 'pretty_formula', 'e_above_hull', 'spacegroup.number', 'cif']
mp = pd.concat([pd.read_csv(f'{a.data}/{s}.csv', usecols=cols).assign(split=s) for s in ('train', 'val', 'test')])
if os.path.exists(CACHE):
    tp = pd.read_csv(CACHE)
else:
    with Pool(a.nproc) as pool:
        res = []
        for i, r in enumerate(pool.imap(template, zip(mp.material_id, mp.cif), chunksize=64), 1):
            res.append(r)
            if i % 5000 == 0:
                print(f'шаблоны: {i} из {len(mp)}', flush=True)
    tp = pd.DataFrame(res, columns=['material_id', 'sg', 'template', 'anon_template', 'n_atoms'])
    tp.to_csv(CACHE, index=False)
df = mp.drop(columns='cif').merge(tp, on='material_id').dropna(subset=['template'])
print(f'структур с шаблоном: {len(df)} из {len(mp)}; разных шаблонов {df.template.nunique()}, '
      f'анонимных {df.anon_template.nunique()}')
CIF = mp.set_index('material_id').cif


def struct(mid):
    return CifParser.from_str(CIF[mid]).parse_structures(primitive=True)[0]


SM = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

# 2. составы с несовпадающими полиморфами одного шаблона
groups = []
for (f, tpl), grp in df.groupby(['pretty_formula', 'template']):
    if len(grp) < 2:
        continue
    ss = [struct(m) for m in grp.material_id]
    clusters = []                      # индексы представителей
    lab = []
    for i, s in enumerate(ss):
        for c, rep in enumerate(clusters):
            if SM.fit(s, ss[rep]):
                lab.append(c)
                break
        else:
            clusters.append(i)
            lab.append(len(clusters) - 1)
    if len(clusters) >= 2:
        g2 = grp.assign(cluster=lab).reset_index(drop=True)
        groups.append((f, tpl, g2, [ss[i] for i in clusters]))
print(f'составов с полиморфами одного шаблона, не совпадающими по StructureMatcher: {len(groups)}')

# 3. частота раскладок в обучающей части по анонимному шаблону
train = df[df.split == 'train']
by_anon = {k: v.material_id.tolist() for k, v in train.groupby('anon_template')}
rng = np.random.default_rng(0)
rows = []
for f, tpl, grp, reps in groups:
    anon = grp.anon_template.iloc[0]
    cand = [m for m in by_anon.get(anon, []) if m not in set(grp.material_id)]
    capped = len(cand) > a.max_train
    if capped:
        cand = list(rng.choice(cand, a.max_train, replace=False))
    counts = np.zeros(len(reps), int)
    other = 0
    for m in cand:
        s = struct(m)
        hit = [SM.fit_anonymous(s, r) for r in reps]
        if sum(hit) == 1:
            counts[hit.index(True)] += 1
        else:
            other += 1                  # не отнесена ни к одной или подходит к нескольким
    for c in range(len(reps)):
        members = grp[grp.cluster == c].sort_values('e_above_hull')
        best = members.iloc[0]
        rows.append({'formula': f, 'template': tpl, 'anon_template': anon, 'cluster': c, 'n_clusters': len(reps),
                     'material_id': best.material_id, 'members': ' '.join(members.material_id),
                     'splits': ' '.join(members.split), 'e_above_hull': best.e_above_hull, 'n_atoms': int(best.n_atoms),
                     'train_same_anon': len(cand), 'train_capped': capped, 'train_this_arrangement': int(counts[c]),
                     'train_unassigned': other})
out = pd.DataFrame(rows)
out.to_csv(f'{R}/template_degeneracy.csv', index=False)

# сводка: для каждой группы — порядок по энергии и по частоте
print('\nсостав | шаблон | раскладка: material_id (части MP-20), энергия над оболочкой мэВ/атом, '
      'в обучении той же раскладки / всего с тем же анонимным шаблоном')
summ = []
for (f, tpl), g2 in out.groupby(['formula', 'template']):
    g2 = g2.sort_values('e_above_hull')
    lo, hi = g2.iloc[0], g2.iloc[-1]
    if lo.train_this_arrangement > hi.train_this_arrangement:
        rel = 'частота и энергия согласны'
    elif lo.train_this_arrangement < hi.train_this_arrangement:
        rel = 'ЧАСТОТА ПРОТИВ ЭНЕРГИИ'
    else:
        rel = 'частоты равны'
    dE = (hi.e_above_hull - lo.e_above_hull) * 1000
    summ.append((rel, dE, f, tpl, g2))
order = {'ЧАСТОТА ПРОТИВ ЭНЕРГИИ': 0, 'частоты равны': 1, 'частота и энергия согласны': 2}
for rel, dE, f, tpl, g2 in sorted(summ, key=lambda x: (order[x[0]], -x[1])):
    print(f'\n{f} | {tpl} | {rel}, разность энергий {dE:.1f} мэВ/атом')
    for r in g2.itertuples():
        print(f'   {r.material_id} ({r.splits}) {r.e_above_hull * 1000:6.1f}  обучение: '
              f'{r.train_this_arrangement} / {r.train_same_anon}{" (подвыборка)" if r.train_capped else ""}')
print('\nитого по группам:', pd.Series([s[0] for s in summ]).value_counts().to_dict())
