"""Прогон CSP-бенчмарка одним вызовом sample.py (модель грузится один раз). Набор выбирается параметром --set.

Команда Python для sample.py берётся из переменной DIFFCSP_PY (по умолчанию «conda run --no-capture-output -n diffcsp_pp
python», как в WSL); в контейнере, где окружение одно, достаточно DIFFCSP_PY=python. Карта выбирается снаружи через
CUDA_VISIBLE_DEVICES: sample.py сам переходит на GPU, если CUDA доступна.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--set', default='v0')
ap.add_argument('--tag', required=True)
ap.add_argument('--per-bin', type=int, default=4)
ap.add_argument('--offset', type=int, default=0)
ap.add_argument('--k', type=int, default=1)
ap.add_argument('--batch-size', type=int, default=20)
ap.add_argument('--chunk-size', type=int, default=0, help='если > 0: брать часть номер --chunk по chunk-size целей, без деления по корзинам')
ap.add_argument('--chunk', type=int, default=0)
ap.add_argument('--ckpt', default='models/diffcsp_pp/checkpoints/pretrained_models/mp_csp')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()

meta = pd.read_csv(f'analysis/results/bench_{a.set}_meta.csv')
targets = {t['label']: t for t in json.load(open(f'configs/csp_targets/bench_{a.set}.json'))}
if a.chunk_size > 0:      # сквозные части по chunk_size целей в случайном порядке: оборванный прогон остаётся несмещённой подвыборкой
    meta = meta.sample(frac=1, random_state=0).reset_index(drop=True)
    sel = meta.iloc[a.chunk * a.chunk_size:(a.chunk + 1) * a.chunk_size]
else:
    sel = meta.groupby('bin', sort=False, group_keys=False).apply(lambda g: g.iloc[a.offset:a.offset + a.per_bin])
if sel.empty:
    print('в этой части нет целей'); sys.exit(0)
entries, order = [], []
for _, r in sel.iterrows():
    t = targets[r['material_id']]
    e = {k: t[k] for k in ('atom_types', 'spacegroup_number', 'wyckoff_letters')}
    for k in range(a.k):
        entries.append(e)
        order.append({'material_id': r['material_id'], 'k': k})
out = os.path.abspath(f'outputs/bench/{a.tag}')
os.makedirs(out, exist_ok=True)
json.dump(entries, open(f'{out}/targets.json', 'w'))
json.dump(order, open(f'{out}/order.json', 'w'))
open(f'{out}/set.txt', 'w').write(a.set)
print(f'набор {a.set}: {len(entries)} структур, батч {a.batch_size}; вывод: {out}', flush=True)
if a.dry_run:
    sys.exit(0)
py = shlex.split(os.environ.get('DIFFCSP_PY', 'conda run --no-capture-output -n diffcsp_pp python'))
cmd = py + ['-u', 'scripts/sample.py',
       '--model_path', os.path.abspath(a.ckpt), '--save_path', out,
       '--json_file', f'{out}/targets.json', '--batch_size', str(a.batch_size)]
sys.exit(subprocess.call(cmd, cwd='models/diffcsp_pp'))
