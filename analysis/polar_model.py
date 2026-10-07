"""Полярные цели: отличается ли успех условной генерации при равной сложности шаблона и частоте прототипа."""
import glob
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

POLAR = set([1] + list(range(3, 10)) + list(range(25, 47)) + list(range(75, 81)) + list(range(99, 111)) +
            list(range(143, 147)) + list(range(156, 162)) + list(range(168, 174)) + list(range(183, 187)))
df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('outputs/bench/*/eval.csv'))]).drop_duplicates(['material_id', 'k'])
df = df[df.generated & df.aligned.astype(bool)].copy()
df = df.merge(pd.read_csv('analysis/results/bench_v0_covariates.csv'), on='material_id', how='left')
df['polar'] = df.sg.isin(POLAR).astype(int)
df['ehull_10meV'] = df.e_above_hull / 0.01
df['log_sg_train'] = np.log1p(df.sg_train_count)
df['log_proto_train'] = np.log1p(df.proto_train_count)
CTRL = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train']
for c in ('match_std', 'match_strict'):
    df[c] = df[c].astype(int)
print(f"целей {len(df)}, полярных {int(df.polar.sum())}")
print(df.groupby('polar')[['match_std', 'match_strict', 'n_free', 'e_above_hull', 'proto_train_count']].median().round(3))
print(df.groupby('polar')[['match_std', 'match_strict']].mean().round(3))

def model(d, y, cols, label):
    cols = [c for c in cols if d[c].std() > 0]
    z = d.copy()
    for c in cols:
        if c not in ('polar', 'ehull_10meV'):
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    try:
        m = sm.Logit(z[y], sm.add_constant(z[cols])).fit(disp=0)
    except Exception as e:
        print(f"\n{label}: не оценивается ({type(e).__name__})")
        return
    ci = m.conf_int()
    print(f"\n--- {label}: {y}, N={int(m.nobs)} (событий {int(d[y].sum())}) ---")
    for c in cols:
        print(f"  {c:<16} OR={np.exp(m.params[c]):7.3f} [{np.exp(ci.loc[c, 0]):7.3f}, {np.exp(ci.loc[c, 1]):7.3f}] p={m.pvalues[c]:.4f}")

for y in ('match_std', 'match_strict'):
    model(df, y, ['polar', 'ehull_10meV'] + CTRL, 'все цели, полярность с контролями')
    model(df[df.n_free >= 1], y, ['polar', 'ehull_10meV'] + CTRL, 'n_free>=1, полярность с контролями')
pol = df[df.polar == 1]
model(pol, 'match_std', ['ehull_10meV', 'n_free', 'log_proto_train'], 'только полярные: энергия')
print("\nполярные по корзинам e_above_hull:")
print(pol.groupby('bin', sort=False).agg(N=('match_std', 'size'), match_std=('match_std', 'mean'), n_free=('n_free', 'median')).round(2))
