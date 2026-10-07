"""Анализ бенчмарка: match rate по корзинам e_above_hull и логистическая регрессия с контролями."""
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

frames = []
for p in sorted(glob.glob('outputs/bench/*/eval.csv')):
    d = pd.read_csv(p)
    d['chunk'] = p.split('/')[-2]
    frames.append(d)
df = pd.concat(frames).drop_duplicates(['material_id', 'k'])
df = df[df.generated & df.aligned.astype(bool)].copy()
cov = pd.read_csv('analysis/results/bench_v0_covariates.csv')[['material_id', 'n_free']]
df = df.merge(cov, on='material_id', how='left')
df['ehull_10meV'] = df.e_above_hull / 0.01
df['log_sg_train'] = np.log1p(df.sg_train_count)
for c in ('match_std', 'match_strict', 'sg_ok_0.01', 'sg_ok_0.1'):
    df[c] = df[c].astype(int)
print(f"структур в анализе: {len(df)} (чанки: {sorted(df.chunk.unique())})")

print("\n=== доля по корзинам e_above_hull (95% ДИ Вильсона) ===")
for b, g in df.groupby('bin', sort=False):
    row = [f"{b:<12} N={len(g):>3}"]
    for c in ('match_std', 'sg_ok_0.1'):
        lo, hi = proportion_confint(g[c].sum(), len(g), method='wilson')
        row.append(f"{c}: {g[c].mean():.2f} [{lo:.2f}, {hi:.2f}]")
    print('  '.join(row))

def logit(y, cols, title):
    z = df.copy()
    for c in cols:
        if c != 'ehull_10meV':
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    m = sm.Logit(z[y], sm.add_constant(z[cols])).fit(disp=0)
    ci = m.conf_int()
    print(f"\n--- {title}: {y} ~ {' + '.join(cols)} (N={int(m.nobs)}) ---")
    for c in cols:
        print(f"{c:<14} OR={np.exp(m.params[c]):.3f}  95% ДИ [{np.exp(ci.loc[c, 0]):.3f}, {np.exp(ci.loc[c, 1]):.3f}]  p={m.pvalues[c]:.4f}")
    return m

CTRL = ['log_sg_train', 'n_atoms', 'n_free']
for y in ('match_std', 'sg_ok_0.1'):
    logit(y, ['ehull_10meV'], 'без контролей')
    logit(y, ['ehull_10meV'] + CTRL, 'с контролями')

mm = df[df.match_std == 1].copy()
mm['log_rmse'] = np.log(mm.rmse_std)
for c in CTRL:
    mm[c + '_z'] = (mm[c] - mm[c].mean()) / mm[c].std()
ols = sm.OLS(mm.log_rmse, sm.add_constant(mm[['ehull_10meV'] + [c + '_z' for c in CTRL]])).fit()
print(f"\n--- log RMSE среди совпавших (N={int(ols.nobs)}) ---")
print(ols.summary().tables[1])
