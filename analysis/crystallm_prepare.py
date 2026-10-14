"""Запросы CrystaLLM для набора целей бенчмарка (штатный конвейер авторов); запросы режутся на части для возобновляемой генерации."""
import argparse
import io
import os
import subprocess
import tarfile
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--set', required=True)
ap.add_argument('--chunk', type=int, default=50)
ap.add_argument('--no-sg', action='store_true', help='запросы только с составом')
ap.add_argument('--repo', default='/home/bebra/matgen/external/CrystaLLM')
ap.add_argument('--py', default='/home/bebra/miniconda3/envs/crystallm/bin/python')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()

D = 'models/diffcsp_pp/data/mp_20'
meta = pd.read_csv(f'analysis/results/bench_{a.set}_meta.csv')
if os.environ.get('SHUFFLE'):      # случайный порядок: частично готовый прогон остаётся несмещённой подвыборкой
    meta = meta.sample(frac=1, random_state=0).reset_index(drop=True)
splits = sorted(set(meta['split'])) if 'split' in meta else ['test']
src = pd.concat([pd.read_csv(f'{D}/{s}.csv', usecols=['material_id', 'cif']) for s in splits]).set_index('material_id')['cif']
tag = a.set + ('_nosg' if a.no_sg else '_sg')
out = os.path.abspath(f'outputs/bench_crystallm/{tag}')
os.makedirs(out, exist_ok=True)
pd.DataFrame({'material_id': meta.material_id, 'cif': [src.loc[m] for m in meta.material_id]}).to_csv(f'{out}/subset.csv', index=False)

steps = [
    [a.py, 'bin/prepare_csv_benchmark.py', f'{out}/subset.csv', f'{out}/orig.tar.gz'],
    [a.py, 'bin/tar_to_pickle.py', f'{out}/orig.tar.gz', f'{out}/orig.pkl.gz'],
    [a.py, 'bin/preprocess.py', f'{out}/orig.pkl.gz', '--out', f'{out}/prep.pkl.gz', '--workers', '4'],
    [a.py, 'bin/make_prompts.py', f'{out}/prep.pkl.gz', '-o', f'{out}/prompts.tar.gz'] + ([] if a.no_sg else ['--with-spacegroup']),
]
for s in steps:
    print(' '.join(s), flush=True)
    if not a.dry_run:
        subprocess.check_call(s, cwd=a.repo)
if a.dry_run:
    print('dry-run: команды не выполнялись')

if os.path.exists(f'{out}/prompts.tar.gz'):
    with tarfile.open(f'{out}/prompts.tar.gz', 'r:gz') as t:
        members = [(m, t.extractfile(m).read()) for m in t.getmembers() if m.isfile()]
    order = {m: i for i, m in enumerate(meta.material_id)}
    members.sort(key=lambda x: order.get(os.path.basename(x[0].name)[:-4], 10**9))
    missing = set(meta.material_id) - {os.path.basename(m.name)[:-4] for m, _ in members}
    n = 0
    for i in range(0, len(members), a.chunk):
        with tarfile.open(f'{out}/prompts_{i // a.chunk:02d}.tar.gz', 'w:gz') as t:
            for m, data in members[i:i + a.chunk]:
                t.addfile(m, io.BytesIO(data))
        n += 1
    print(f'запросов {len(members)} из {len(meta)} (не получили запрос: {len(missing)}); частей по {a.chunk}: {n}')
    print('пример запроса:\n' + members[0][1].decode()[:500])
