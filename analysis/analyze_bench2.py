"""Расширенный анализ бенчмарка v0: нетривиальные цели, страты по сложности шаблона, строгий матчер, RMSE."""
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

frames = [pd.read_csv(p) for p in sorted(glob.glob('outputs/bench/*/eval.csv'))]
df = pd.concat(frames).drop_duplicates(['material_id', 'k'])
df = df[df.generated & df.aligned.astype(bool)].copy()
df = df.merge(pd.read_csv('analysis/results/bench_v0_covariates.csv'), on='material_id', how='left')
df['ehull_10meV'] = df.e_above_hull / 0.01
df['log_sg_train'] = np.log1p(df.sg_train_count)
CTRL = ['log_sg_train', 'n_atoms', 'n_free']
if 'proto_train_count' in df:
    df['log_proto_train'] = np.log1p(df.proto_train_count)
    CTRL.append('log_proto_train')
for c in ('match_std', 'match_strict', 'sg_ok_0.01', 'sg_ok_0.1'):
    df[c] = df[c].astype(int)
print(f"структур: {len(df)}; контроли: {CTRL}")

print("\n=== строгий и стандартный матчер по корзинам ===")
print(df.groupby('bin', sort=False)[['match_std', 'match_strict']].mean().round(2))

print("\n=== тривиальные цели: число свободных параметров ===")
nf = pd.cut(df.n_free, [-1, 0, 1, 2, 4, 8, 100], labels=['0', '1', '2', '3-4', '5-8', '9+'])
print(df.groupby(nf, observed=True).agg(N=('match_std', 'size'), match_std=('match_std', 'mean'),
                                        ehull_med=('e_above_hull', 'median')).round(3))
print("n_free=0 по корзинам:", df[df.n_free == 0].groupby('bin', sort=False).size().to_dict())

def fit(d, y, cols):
    z = d.copy()
    cs = [c for c in cols if c == 'ehull_10meV' or z[c].std() > 0]
    for c in cs:
        if c != 'ehull_10meV':
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    try:
        m = sm.Logit(z[y], sm.add_constant(z[cs])).fit(disp=0)
    except Exception:
        return None
    ci = m.conf_int().loc['ehull_10meV']
    return np.exp(m.params['ehull_10meV']), np.exp(ci[0]), np.exp(ci[1]), m.pvalues['ehull_10meV']

def line(d, y, label):
    r0 = fit(d, y, ['ehull_10meV'])
    r1 = fit(d, y, ['ehull_10meV'] + CTRL)
    f = lambda r: 'не оценивается' if r is None else f"OR={r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] p={r[3]:.3f}"
    print(f"{label:<26} N={len(d):>3} доля={d[y].mean():.2f} | без контр.: {f(r0)} | с контр.: {f(r1)}")

for y in ('match_std', 'match_strict', 'sg_ok_0.1'):
    print(f"\n=== OR на +10 мэВ/атом e_above_hull, исход {y} ===")
    line(df, y, 'все цели')
    for thr in (1, 2, 3):
        line(df[df.n_free >= thr], y, f'n_free >= {thr}')
    q = pd.qcut(df.n_free, 3, duplicates='drop')
    for lvl, g in df.groupby(q, observed=True):
        line(g, y, f'n_free {lvl}')

print("\n=== RMSE среди совпавших (нулевой RMSE = структура целиком задана шаблоном) ===")
mm = df[df.match_std == 1]
print(f"совпавших {len(mm)}, из них RMSE=0: {(mm.rmse_std <= 1e-9).sum()}")
nz = mm[mm.rmse_std > 1e-9].copy()
rho, p = spearmanr(nz.e_above_hull, nz.rmse_std)
print(f"Спирмен(e_above_hull, RMSE), N={len(nz)}: rho={rho:.3f}, p={p:.4f}")
nz['log_rmse'] = np.log(nz.rmse_std)
cols = ['ehull_10meV'] + CTRL
for c in CTRL:
    nz[c] = (nz[c] - nz[c].mean()) / nz[c].std()
ols = sm.OLS(nz.log_rmse, sm.add_constant(nz[cols])).fit()
print(ols.summary().tables[1])
