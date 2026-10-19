"""Перекрёстное сопоставление: что именно выдала модель на каждую попытку.

Каждая генерация сравнивается не только со своей целью, но и со всеми полиморфами того же состава во всём MP-20
(train, val, test). Итог по попытке — одна категория:
  цель                      — совпала с целью (строгие допуски 0,2; 0,3; 5°)
  другой полиморф           — совпала (строго) с другим известным полиморфом того же состава
  цель (станд.)             — со своей целью совпала только при стандартных допусках 0,3; 0,5; 10°
  другой полиморф (станд.)  — с другим полиморфом совпала только при стандартных допусках
  новая структура           — состав верный, но ни с одним известным полиморфом не совпала
  другой состав / невалидный CIF

Источники генераций:
  --tar  архив(ы) CIF с именами <material_id>__<k>.cif (готовые генерации CrystaLLM или post_*.tar.gz)
  --bench-dir  папка прогона DiffCSP++ (run_bench.py: order.json + <номер>.cif)

Использование:
  python analysis/cross_match.py --set pairs_v1 --tar ПУТЬ/gen_mp_20_small.tar.gz --out analysis/results/cross_pairs_v1_pub.csv
  python analysis/cross_match.py --set pairs_v1 --tar "outputs/bench_crystallm/pairs_v1_sg/post_*.tar.gz" --out ...
  python analysis/cross_match.py --set pairs_v1nf --bench-dir "outputs/bench/pv1_o*" --out ...
"""
import argparse
import glob
import json
import os
import tarfile
import warnings
from functools import lru_cache
from multiprocessing import Pool
import pandas as pd
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--set', required=True)
ap.add_argument('--tar', nargs='*', default=[])
ap.add_argument('--bench-dir', nargs='*', default=[])
ap.add_argument('--out', required=True)
ap.add_argument('--nproc', type=int, default=6)
a = ap.parse_args()

D = 'models/diffcsp_pp/data/mp_20'
meta = pd.read_csv(f'analysis/results/bench_{a.set}_meta.csv').set_index('material_id')
cols = ['material_id', 'pretty_formula', 'e_above_hull', 'spacegroup.number', 'cif']
allmp = pd.concat([pd.read_csv(f'{D}/{s}.csv', usecols=cols).assign(split=s) for s in ('train', 'val', 'test')])
allmp = allmp[allmp.pretty_formula.isin(set(meta.formula))]
POLY = {f: grp[['material_id', 'e_above_hull', 'spacegroup.number', 'split', 'cif']].to_dict('records')
        for f, grp in allmp.groupby('pretty_formula')}
FORMULA, SG = meta.formula.to_dict(), meta.sg.to_dict()
MT = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
MS = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)


@lru_cache(maxsize=64)
def refs(formula):
    out = []
    for r in POLY.get(formula, []):
        s = CifParser.from_str(r['cif']).parse_structures(primitive=True)[0]
        out.append((r['material_id'], r['e_above_hull'], int(r['spacegroup.number']), r['split'], s))
    return out


def classify(item):
    mid, k, kind, payload = item
    warnings.filterwarnings('ignore')
    row = {'material_id': mid, 'k': k, 'category': 'невалидный CIF', 'matched_id': None, 'matched_E': None,
           'matched_sg': None, 'matched_split': None, 'gen_sg_0.1': None}
    try:
        gen = (CifParser.from_str(payload).parse_structures(primitive=True)[0] if kind == 'cif_text'
               else Structure.from_file(payload))
    except Exception:
        return row
    try:
        row['gen_sg_0.1'] = SpacegroupAnalyzer(gen, symprec=0.1).get_space_group_number()
    except Exception:
        row['gen_sg_0.1'] = -1
    f = FORMULA[mid]
    if gen.composition.reduced_formula != Composition(f).reduced_formula:
        row['category'] = 'другой состав'
        return row
    cand = refs(f)
    cand = sorted(cand, key=lambda c: c[0] != mid)          # своя цель — первой
    for tol, M in (('', MT), (' (станд.)', MS)):
        for cid, e, sg, split, s in cand:
            if M.fit(gen, s):
                row.update(category=('цель' if cid == mid else 'другой полиморф') + tol, matched_id=cid,
                           matched_E=e, matched_sg=sg, matched_split=split)
                return row
    row['category'] = 'новая структура'
    return row


def split_name(name):
    b = os.path.basename(name)
    b = b[:-4] if b.endswith('.cif') else b
    mid, k = (b.rsplit('__', 1) if '__' in b else b.rsplit('_', 1))
    return mid, int(k)


def items():
    for pattern in a.tar:
        for path in sorted(glob.glob(pattern)):
            with tarfile.open(path, 'r:*') as tar:
                for m in tar:
                    if not m.isfile() or not m.name.endswith('.cif'):
                        continue
                    try:
                        mid, k = split_name(m.name)
                    except ValueError:
                        continue
                    if mid in FORMULA:
                        yield mid, k, 'cif_text', tar.extractfile(m).read().decode()
    for pattern in a.bench_dir:
        for d in sorted(glob.glob(pattern)):
            order = json.load(open(os.path.join(d, 'order.json')))
            cifs = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(d, '*.cif'))
                    if os.path.basename(p)[:-4].isdigit()}
            if not cifs:
                continue
            off = min(cifs)
            for i, o in enumerate(order):
                p = cifs.get(i + off)
                if p is not None and o['material_id'] in FORMULA:
                    yield o['material_id'], o['k'], 'cif_file', p


with Pool(a.nproc) as pool:
    rows = []
    for i, r in enumerate(pool.imap(classify, items(), chunksize=8), 1):
        rows.append(r)
        if i % 1000 == 0:
            print(f'обработано {i}', flush=True)
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit('генераций для целей набора не найдено: проверь пути и имена файлов')
df = df.merge(meta[['formula', 'e_above_hull', 'sg', 'pair_role']].reset_index(), on='material_id', how='left')
df.to_csv(a.out, index=False)
print(f'попыток {len(df)}, целей {df.material_id.nunique()} из {len(meta)}; результат: {a.out}')
print(pd.crosstab(df.pair_role, df.category, normalize='index').round(3).to_string())
