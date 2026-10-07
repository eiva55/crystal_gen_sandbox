"""Бенчмарк v0: полные таблицы коэффициентов и анализ недо-искажения (окна допуска symprec)."""
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

frames = [pd.read_csv(p) for p in sorted(glob.glob('outputs/bench/*/eval.csv'))]
df = pd.concat(frames).drop_duplicates(['material_id', 'k'])
df = df[df.generated & df.aligned.astype(bool)].copy()
df = df.merge(pd.read_csv('analysis/results/bench_v0_covariates.csv'), on='material_id', how='left')
df['ehull_10meV'] = df.e_above_hull / 0.01
df['log_sg_train'] = np.log1p(df.sg_train_count)
df['log_proto_train'] = np.log1p(df.proto_train_count)
CTRL = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train']
for c in ('match_std', 'match_strict', 'sg_ok_0.01', 'sg_ok_0.1'):
    df[c] = df[c].astype(int)

def full(d, y, label):
    cols = ['ehull_10meV'] + [c for c in CTRL if d[c].std() > 0]
    z = d.copy()
    for c in cols:
        if c != 'ehull_10meV':
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    try:
        m = sm.Logit(z[y], sm.add_constant(z[cols])).fit(disp=0)
    except Exception as e:
        print(f"\n{label}: не оценивается ({type(e).__name__})")
        return
    ci = m.conf_int()
    print(f"\n--- {label}: {y}, N={int(m.nobs)}, доля={d[y].mean():.2f} ---")
    for c in cols:
        print(f"  {c:<16} OR={np.exp(m.params[c]):7.3f}  [{np.exp(ci.loc[c, 0]):7.3f}, {np.exp(ci.loc[c, 1]):7.3f}]  p={m.pvalues[c]:.4f}")

print("ПОЛНЫЕ МОДЕЛИ (ehull_10meV: OR на +10 мэВ/атом; контроли: OR на +1 СО)")
for y in ('match_std', 'match_strict'):
    full(df, y, 'все цели')
    full(df[df.n_free >= 3], y, 'n_free >= 3')

wp = 'analysis/results/bench_v0_windows.csv'
if os.path.exists(wp):
    w = pd.read_csv(wp)[['material_id', 'k', 'tau_ref', 'tau_gen']]
    d = df.merge(w, on=['material_id', 'k'])
    print(f"\n\nОКНА ДОПУСКА (прокси амплитуды искажения). целей с окном: {len(d)}")
    print(f"  у сэмпла нет целевой группы даже при 0.001 Å: {int(d.tau_gen.isna().sum())}; "
          f"у эталона нет: {int(d.tau_ref.isna().sum())}")
    d = d[(d.n_free >= 1) & d.tau_ref.notna() & d.tau_gen.notna()].copy()
    d['dlog'] = np.log10(d.tau_gen / d.tau_ref)
    d['under'] = (d.tau_gen < 0.5 * d.tau_ref).astype(int)
    d['fragile_ref'] = d.tau_ref < 0.2
    print(f"  n_free>=1 и окна определены: N={len(d)}; хрупкие эталоны (окно < 0.2 Å): {int(d.fragile_ref.sum())}")
    print("\n  по корзинам e_above_hull: доля 'недо-искажённых' сэмплов (окно сэмпла < 0.5 окна эталона), медиана log10(tau_gen/tau_ref)")
    g = d.groupby('bin', sort=False)
    print(pd.DataFrame({'N': g.size(), 'under': g.under.mean().round(2), 'median_dlog': g.dlog.median().round(3),
                        'frac_fragile_ref': g.fragile_ref.mean().round(2)}))
    rho, p = spearmanr(d.e_above_hull, d.dlog)
    print(f"\n  Спирмен(e_above_hull, dlog): rho={rho:.3f}, p={p:.4f}")
    full(d, 'under', 'недо-искажение, все n_free>=1')
    fr = d[d.fragile_ref]
    if len(fr) > 30:
        full(fr, 'under', 'недо-искажение, только хрупкие эталоны')
else:
    print("\n(файл окон не найден — запусти analysis/bench_windows.py)")
