"""Доля успеха по числу структур того же типа в обучающей выборке (покрытие), для test- и train-целей."""
import glob
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

fr = [pd.read_csv(p) for p in sorted(glob.glob('outputs/bench/*/eval.csv'))]
df = pd.concat(fr, ignore_index=True).drop_duplicates(['material_id', 'k'])
df = df[df.generated & df.aligned.astype(bool)].copy()
nb = pd.read_csv('analysis/results/bench_neighbors.csv')[['material_id', 'n_same_type']]
covs = [pd.read_csv(p)[['material_id', 'n_free']] for p in glob.glob('analysis/results/bench_*_covariates.csv')]
for p in glob.glob('analysis/results/bench_*_meta.csv'):
    m = pd.read_csv(p)
    if 'n_free' in m:
        covs.append(m[['material_id', 'n_free']])
cov = pd.concat(covs).drop_duplicates('material_id')
df = df.drop(columns=[c for c in ('n_free', 'n_same_type') if c in df]).merge(nb, on='material_id').merge(cov, on='material_id')
if 'split' not in df:
    df['split'] = 'test'
df['split'] = df['split'].fillna('test')
df = df[df.n_free >= 1]
df['покрытие'] = pd.cut(df.n_same_type, [-1, 0, 3, 15, 60, 10**6], labels=['0', '1-3', '4-15', '16-60', '>60'])
for y in ('match_strict', 'match_std'):
    df[y] = df[y].astype(int)
    print(f"\n=== {y}: доля успеха и число целей по числу структур того же типа в обучающей выборке (цели с n_free>=1) ===")
    t = df.pivot_table(index='покрытие', columns='split', values=y, aggfunc=['mean', 'size'], observed=True)
    t.loc['все'] = [df[df.split == s][y].mean() for s in ('test', 'train')] + [(df.split == s).sum() for s in ('test', 'train')]
    print(t.round(2))
