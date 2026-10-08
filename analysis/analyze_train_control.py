"""Контроль «train против test»: влияет ли принадлежность цели к обучающей выборке на успех при равных ковариатах."""
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
df = df.drop(columns=[c for c in ('n_free', 'proto_train_count') if c in df]).merge(cov, on='material_id', how='left')
if 'split' not in df:
    df['split'] = 'test'
df['split'] = df['split'].fillna('test')
df['T'] = (df.split == 'train').astype(int)
df['ehull_10meV'] = df.e_above_hull / 0.01
df['log_sg_train'] = np.log1p(df.sg_train_count.clip(lower=0))
df['log_proto_train'] = np.log1p(df.proto_train_count.clip(lower=0))
for c in ('match_std', 'match_strict'):
    df[c] = df[c].astype(int)
CTRL = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train']
print(f"целей: test {int((df['T'] == 0).sum())}, train {int((df['T'] == 1).sum())}")
print("\nдоля совпадений (строгие допуски) по корзинам и принадлежности:")
print(df.pivot_table(index='bin', columns='split', values='match_strict', aggfunc='mean', sort=False).round(2))

def fit(z, y, cols):
    cols = [c for c in cols if z[c].std() > 0]
    try:
        return sm.Logit(z[y], sm.add_constant(z[cols])).fit(disp=0)
    except Exception:
        return None

def show(m, name, one_sided=False):
    if m is None or name not in m.params:
        print(f"  {name:<14} не оценивается")
        return
    ci = m.conf_int().loc[name]
    p = m.pvalues[name]
    if one_sided:
        p = p / 2 if m.params[name] > 0 else 1 - p / 2
    print(f"  {name:<14} ОШ={np.exp(m.params[name]):7.3f} [{np.exp(ci[0]):7.3f}; {np.exp(ci[1]):7.3f}] p={p:.4f}{' (односторонний)' if one_sided else ''}")

def report(d, y, label):
    z = d.copy()
    for c in CTRL:
        if z[c].std() > 0:
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    z['ET'] = z.ehull_10meV * z['T']
    print(f"\n=== {label}: {y}, N={len(z)} (test {int((z['T'] == 0).sum())}, train {int((z['T'] == 1).sum())}) ===")
    print("только test, энергия:")
    show(fit(z[z['T'] == 0], y, ['ehull_10meV'] + CTRL), 'ehull_10meV')
    print("только train, энергия:")
    show(fit(z[z['T'] == 1], y, ['ehull_10meV'] + CTRL), 'ehull_10meV')
    print("все цели, принадлежность к train (H1: ОШ > 1):")
    show(fit(z, y, ['ehull_10meV', 'T'] + CTRL), 'T', one_sided=True)
    print("все цели, взаимодействие энергия × train:")
    mm = fit(z, y, ['ehull_10meV', 'T', 'ET'] + CTRL)
    show(mm, 'ehull_10meV'); show(mm, 'ET')

for y in ('match_strict', 'match_std'):
    report(df[df.n_free >= 1], y, 'n_free >= 1')
    report(df[df.n_free >= 3], y, 'n_free >= 3')
