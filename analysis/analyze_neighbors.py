"""Остаётся ли эффект принадлежности к обучающей выборке при учёте ближайших структур (почти дубликатов) в ней."""
import glob
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

frames = [pd.read_csv(p) for p in sorted(glob.glob('outputs/bench/*/eval.csv'))]
df = pd.concat(frames, ignore_index=True).drop_duplicates(['material_id', 'k'])
df = df[df.generated & df.aligned.astype(bool)].copy()
covs = [pd.read_csv(p)[['material_id', 'n_free', 'proto_train_count']] for p in glob.glob('analysis/results/bench_*_covariates.csv')]
for p in glob.glob('analysis/results/bench_*_meta.csv'):
    m = pd.read_csv(p)
    if 'n_free' in m:
        covs.append(m[['material_id', 'n_free', 'proto_train_count']])
cov = pd.concat(covs).drop_duplicates('material_id')
nb = pd.read_csv('analysis/results/bench_neighbors.csv')[['material_id', 'n_same_chem', 'n_same_type', 'capped']]
df = df.drop(columns=[c for c in ('n_free', 'proto_train_count') if c in df]).merge(cov, on='material_id', how='left').merge(nb, on='material_id', how='left')
if 'split' not in df:
    df['split'] = 'test'
df['split'] = df['split'].fillna('test')
df['T'] = (df.split == 'train').astype(int)
df['ehull_10meV'] = df.e_above_hull / 0.01
df['log_sg_train'] = np.log1p(df.sg_train_count.clip(lower=0))
df['log_proto_train'] = np.log1p(df.proto_train_count.clip(lower=0))
df['log_same_type'] = np.log1p(df.n_same_type)
df['has_chem'] = (df.n_same_chem > 0).astype(int)
for c in ('match_std', 'match_strict'):
    df[c] = df[c].astype(int)
CTRL = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train', 'log_same_type']
CONT = CTRL
print(f"целей: test {int((df['T'] == 0).sum())}, train {int((df['T'] == 1).sum())}")
print("\nдоли целей с соседями в обучающей выборке (для train-целей сама цель исключена):")
print(df.groupby('split').agg(same_type=('n_same_type', lambda x: (x > 0).mean()), same_chem=('has_chem', 'mean'),
                              median_n_same_type=('n_same_type', 'median'), capped=('capped', 'mean')).round(3))
print("\nдоля совпадений (строгие допуски) по наличию почти дубликата той же химии:")
print(df.pivot_table(index='has_chem', columns='split', values='match_strict', aggfunc=['mean', 'size']).round(2))

def fit(z, y, cols):
    cols = [c for c in cols if z[c].std() > 0]
    try:
        return sm.Logit(z[y], sm.add_constant(z[cols])).fit(disp=0)
    except Exception:
        return None

def show(m, names, one_sided=()):
    for name in names:
        if m is None or name not in m.params:
            print(f"  {name:<16} не оценивается")
            continue
        ci = m.conf_int().loc[name]
        p = m.pvalues[name]
        tag = ''
        if name in one_sided:
            p = p / 2 if m.params[name] > 0 else 1 - p / 2
            tag = ' (односторонний)'
        print(f"  {name:<16} ОШ={np.exp(m.params[name]):7.3f} [{np.exp(ci[0]):7.3f}; {np.exp(ci[1]):7.3f}] p={p:.4f}{tag}")

def report(d, y, label):
    z = d.copy()
    for c in CONT:
        if z[c].std() > 0:
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    base = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train']
    print(f"\n=== {label}: {y}, N={len(z)} (test {int((z['T'] == 0).sum())}, train {int((z['T'] == 1).sum())}) ===")
    print("без учёта соседей (как раньше):")
    show(fit(z, y, ['ehull_10meV', 'T'] + base), ['ehull_10meV', 'T'], one_sided=('T',))
    print("с учётом соседей (число структур того же типа и почти дубликат той же химии):")
    show(fit(z, y, ['ehull_10meV', 'T', 'has_chem'] + CTRL), ['ehull_10meV', 'T', 'log_same_type', 'has_chem'], one_sided=('T',))
    print("только тестовая часть, влияние соседей на невиденные цели:")
    show(fit(z[z['T'] == 0], y, ['ehull_10meV', 'has_chem'] + CTRL), ['ehull_10meV', 'log_same_type', 'has_chem'])

for y in ('match_strict', 'match_std'):
    report(df[df.n_free >= 1], y, 'n_free >= 1')
    report(df[df.n_free >= 3], y, 'n_free >= 3')
