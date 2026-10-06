"""Есть ли целевые фазы HfO2 в MP-20 и как часто встречаются группы."""
import pandas as pd
from pathlib import Path

IDS = ['mp-352', 'mp-685097']
df = pd.read_csv('models/adit/data/mp_20/raw/all.csv',
                 usecols=['material_id', 'pretty_formula', 'spacegroup.number', 'e_above_hull'])
print(f"all.csv: {len(df)} строк")
print("\nЦелевые id:")
print(df[df.material_id.isin(IDS)])
print("\nВсе HfO2 в MP-20:")
print(df[df.pretty_formula == 'HfO2'].sort_values('e_above_hull').to_string())
vc = df['spacegroup.number'].value_counts()
print("\nЧастота групп:")
for g in [14, 29, 60, 61, 160, 205, 225, 227]:
    print(f"  SG{g}: {vc.get(g, 0)} ({100 * vc.get(g, 0) / len(df):.2f}%)")
print("\ne_above_hull (весь MP-20):")
print(df.e_above_hull.describe())

d = Path('models/diffcsp_pp/data/mp_20')
print('\nmodels/diffcsp_pp/data/mp_20:',
      sorted(p.name for p in d.glob('*')) if d.exists() else 'нет папки')
for name in ['train', 'val', 'test']:
    p = d / f'{name}.csv'
    if p.exists():
        ids = set(pd.read_csv(p, usecols=['material_id']).material_id)
        print(f"{name}: {len(ids)} структур, {({i: (i in ids) for i in IDS})}")
