"""Недо-искажение (окно допуска сэмпла против эталона) для полярных и неполярных целей."""
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
warnings.filterwarnings('ignore')

POLAR = set([1] + list(range(3, 10)) + list(range(25, 47)) + list(range(75, 81)) + list(range(99, 111)) +
            list(range(143, 147)) + list(range(156, 162)) + list(range(168, 174)) + list(range(183, 187)))
df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('outputs/bench/*/eval.csv'))]).drop_duplicates(['material_id', 'k'])
df = df.merge(pd.read_csv('analysis/results/bench_v0_covariates.csv'), on='material_id', how='left')
w = pd.read_csv('analysis/results/bench_v0_windows.csv')[['material_id', 'k', 'tau_ref', 'tau_gen']]
d = df.merge(w, on=['material_id', 'k'])
d = d[(d.n_free >= 1) & d.tau_ref.notna() & d.tau_gen.notna()].copy()
d['polar'] = d.sg.isin(POLAR)
d['under'] = (d.tau_gen < 0.5 * d.tau_ref).astype(int)
d['dlog'] = np.log10(d.tau_gen / d.tau_ref)
print(f"целей с окнами и n_free>=1: {len(d)}; полярных: {int(d.polar.sum())}")
print(d.groupby('polar').agg(N=('under', 'size'), under=('under', 'mean'), median_dlog=('dlog', 'median'),
                             match_std=('match_std', 'mean')).round(3))
a = int((d.polar & (d.under == 1)).sum()); b = int((d.polar & (d.under == 0)).sum())
c = int((~d.polar & (d.under == 1)).sum()); e = int((~d.polar & (d.under == 0)).sum())
print(f"\nполярные: недо-искажено {a}, нет {b}; неполярные: недо-искажено {c}, нет {e}")
print(f"точный критерий Фишера (полярные чаще недо-искажены): p={fisher_exact([[a, b], [c, e]], alternative='greater')[1]:.4f}")
p = d[d.polar]
print("\nполярные цели: по e_above_hull")
print(p.groupby(pd.cut(p.e_above_hull, [-1, 0.02, 0.04, 0.081]), observed=True).agg(
    N=('under', 'size'), under=('under', 'mean'), median_dlog=('dlog', 'median')).round(3))
print("\nполярные цели с недо-искажением:")
print(p[p.under == 1][['material_id', 'formula', 'sg', 'e_above_hull', 'tau_ref', 'tau_gen', 'match_std']].round(3).to_string())
