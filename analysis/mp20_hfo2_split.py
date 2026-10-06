"""Где лежат все HfO2 из MP-20 (train/val/test) и сколько метастабильных точек в test."""
import pandas as pd

IDS = ['mp-352', 'mp-776532', 'mp-685097', 'mp-1018721']
keep = lambda c: c in ('material_id', 'spacegroup.number', 'e_above_hull', 'pretty_formula')
sp = {n: pd.read_csv(f'models/diffcsp_pp/data/mp_20/{n}.csv', usecols=keep) for n in ['train', 'val', 'test']}
allv = pd.concat([d.assign(split=n) for n, d in sp.items()])
print(allv[allv.material_id.isin(IDS)].to_string())

if 'spacegroup.number' in sp['train']:
    tr = sp['train']['spacegroup.number'].value_counts()
    print("\nЧисло структур группы в train:", {g: int(tr.get(g, 0)) for g in [14, 29, 60, 61, 136, 137, 160, 205, 227]})
te = sp['test']
for t in [0.01, 0.02, 0.04, 0.06]:
    print(f"test: e_above_hull >= {t}: {(te.e_above_hull >= t).sum()} из {len(te)}")
g = te.groupby('pretty_formula').size()
print("составов в test с >=2 структурами:", int((g >= 2).sum()), ", >=3:", int((g >= 3).sum()))
