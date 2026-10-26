"""Подмножество готового набора целей по списку составов (для пробных прогонов всего пути: генерация — сопоставление — разбор).

Пишет analysis/results/bench_<dst>_meta.csv и configs/csp_targets/bench_<dst>.json в том же формате, что и исходный набор.

Использование:
  python analysis/subset_set.py --src pairs_v1nf --dst pairs_smoke --formulas Ga HfIr NbS2
"""
import argparse
import json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--formulas', nargs='+', required=True)
a = ap.parse_args()

meta = pd.read_csv(f'analysis/results/bench_{a.src}_meta.csv')
sub = meta[meta.formula.isin(a.formulas)]
missing = set(a.formulas) - set(sub.formula)
if missing:
    raise SystemExit(f'нет в наборе {a.src}: {sorted(missing)}')
targets = [t for t in json.load(open(f'configs/csp_targets/bench_{a.src}.json')) if t['label'] in set(sub.material_id)]
assert len(targets) == len(sub), 'meta и json исходного набора не согласованы'
sub.to_csv(f'analysis/results/bench_{a.dst}_meta.csv', index=False)
json.dump(targets, open(f'configs/csp_targets/bench_{a.dst}.json', 'w'), indent=1)
print(f'{a.dst}: {len(sub)} целей, составов {sub.formula.nunique()}')
print(sub[['material_id', 'formula', 'pair_role', 'sg', 'n_atoms', 'n_free', 'e_above_hull', 'split']].to_string(index=False))
