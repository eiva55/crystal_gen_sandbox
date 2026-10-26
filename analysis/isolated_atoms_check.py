"""Проверка энергий CHGNet для «новых структур»: есть ли в сгенерированной структуре изолированные атомы.

CHGNet строит граф атомов с радиусом 6 Å; атом без соседей в этом радиусе даёт предупреждение
«isolated atom(s) … CHGNet calculation will likely go wrong», и энергия такой структуры ненадёжна.
Скрипт читает те же CIF, что new_structure_energy.py, отмечает попытки с изолированными атомами (до релаксации),
пишет объём на атом относительно цели и пересчитывает медианы без отмеченных попыток.

Попытка задаётся тройкой (source, material_id, k), как в new_structure_energy.py.
Использование (окружение mp_lookup):
  python analysis/isolated_atoms_check.py --energy analysis/results/new_structure_energy_k10.csv \
    --tar "outputs/bench_crystallm/pair_sg/post_*.tar.gz" "outputs/bench_crystallm/pairs_v1_sg/post_*.tar.gz" \
    --out analysis/results/new_structure_energy_k10_isolated.csv > analysis/results/new_structure_energy_k10_isolated.txt
"""
import argparse
import glob
import os
import tarfile
import warnings
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--energy', required=True)
ap.add_argument('--tar', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--cutoff', type=float, default=6.0)
ap.add_argument('--data', default='models/diffcsp_pp/data/mp_20')
a = ap.parse_args()

en = pd.read_csv(a.energy)
if 'source' not in en:
    if en.duplicated(['material_id', 'k']).any():
        raise SystemExit('в файле энергий повторяются (цель, k) и нет столбца source: пересчитай new_structure_energy.py')
    en['source'] = ''
want = set(zip(en.source, en.material_id, en.k))


def split_name(name):
    b = os.path.basename(name)
    b = b[:-4] if b.endswith('.cif') else b
    mid, k = b.rsplit('__', 1) if '__' in b else b.rsplit('_', 1)
    return mid, int(k)


gens = {}
for pat in a.tar:
    for path in sorted(glob.glob(pat)):
        with tarfile.open(path, 'r:*') as tar:
            for m in tar:
                if m.isfile() and m.name.endswith('.cif'):
                    try:
                        key = split_name(m.name)
                    except ValueError:
                        continue
                    key = (path if (path, *key) in want else '', *key)
                    if key in want:
                        gens[key] = tar.extractfile(m).read().decode()
print(f'попыток в файле энергий {len(want)}, найдено CIF {len(gens)}')

ref = pd.concat([pd.read_csv(f'{a.data}/{s}.csv', usecols=['material_id', 'cif']) for s in ('train', 'val', 'test')])
ref = ref[ref.material_id.isin(set(en.material_id))].drop_duplicates('material_id').set_index('material_id').cif
vref = {mid: (lambda s: s.volume / len(s))(CifParser.from_str(c).parse_structures(primitive=True)[0]) for mid, c in ref.items()}

rows = []
for r in en.itertuples():
    txt = gens.get((r.source, r.material_id, r.k))
    row = {'source': r.source, 'material_id': r.material_id, 'k': r.k, 'n_isolated': np.nan, 'min_dist': np.nan, 'vol_ratio': np.nan}
    if txt is not None:
        try:
            s = CifParser.from_str(txt).parse_structures(primitive=True)[0]
            nb = s.get_all_neighbors(a.cutoff)
            row['n_isolated'] = sum(len(x) == 0 for x in nb)
            near = [min(n.nn_distance for n in x) for x in nb if x]
            row['min_dist'] = min(near) if near else np.nan
            row['vol_ratio'] = (s.volume / len(s)) / vref.get(r.material_id, np.nan)
        except Exception:
            pass
    rows.append(row)
fl = pd.DataFrame(rows)
df = en.merge(fl, on=['source', 'material_id', 'k'], how='left', validate='one_to_one')
df['isolated'] = df.n_isolated.fillna(0) > 0
df.to_csv(a.out, index=False)

print(f'попыток с изолированными атомами: {int(df.isolated.sum())} из {len(df)} '
      f'(по ролям: {df[df.isolated].pair_role.value_counts().to_dict()})')
print(f'объём на атом относительно цели: медиана {df.vol_ratio.median():.2f}; '
      f'у попыток с изолированными атомами {df[df.isolated].vol_ratio.median():.2f}')


def summary(d, lab):
    d = d.dropna(subset=['d_target'])
    print(f'\n{lab}')
    for role, g in d.groupby('pair_role'):
        comp = g.groupby('formula').apply(lambda x: pd.Series({'t': (x.d_target < 0).mean(), 's': (x.d_stable < 0).mean()}))
        print(f'  {role:7s} попыток {len(g):4d}  составов {g.formula.nunique():3d}  '
              f'медиана E_ген−E_цели {g.d_target.median() * 1000:+7.1f} мэВ/атом  '
              f'медиана E_ген−E_стаб. {g.d_stable.median() * 1000:+7.1f}  '
              f'доля ниже цели {comp.t.mean():.3f}  ниже стабильного {comp.s.mean():.3f}')


summary(df, 'все попытки (как в new_structure_energy_k10.txt)')
summary(df[~df.isolated], 'без попыток с изолированными атомами')
