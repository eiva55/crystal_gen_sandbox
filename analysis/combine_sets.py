"""Объединить наборы целей с непересекающимися составами в один (например, тестовые и валидационные пары).

Пишет analysis/results/bench_<dst>_meta.csv и configs/csp_targets/bench_<dst>.json. Составы наборов не должны
пересекаться, иначе пары одного состава из разных наборов смешаются.

Использование:
  python analysis/combine_sets.py --dst pairs_tv_nf --src pairs_v1nf pairs_val_v1nf
"""
import argparse
import json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--dst', required=True)
ap.add_argument('--src', nargs='+', required=True)
a = ap.parse_args()

metas, targets = [], []
for s in a.src:
    m = pd.read_csv(f'analysis/results/bench_{s}_meta.csv').assign(source_set=s)
    metas.append(m)
    targets += json.load(open(f'configs/csp_targets/bench_{s}.json'))
m = pd.concat(metas, ignore_index=True)
shared = m.groupby('formula').source_set.nunique()
if (shared > 1).any():
    raise SystemExit(f'составы встречаются в нескольких наборах: {sorted(shared[shared > 1].index)[:10]}')
if m.material_id.duplicated().any():
    raise SystemExit('повторяются material_id')
m.to_csv(f'analysis/results/bench_{a.dst}_meta.csv', index=False)
json.dump(targets, open(f'configs/csp_targets/bench_{a.dst}.json', 'w'), indent=1)
print(f'{a.dst}: целей {len(m)}, составов {m.formula.nunique()}; по наборам: {m.source_set.value_counts().to_dict()}')
