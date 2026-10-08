"""Зависимость успеха генерации от энергии над оболочкой на всей тестовой части (опубликованные генерации CrystaLLM).
Модель: логистическая регрессия по попыткам, стандартные ошибки кластеризованы по цели.
Эквивалентность (H1): 90% ДИ отношения шансов на +10 мэВ/атом внутри [0,917; 1,090] (два односторонних критерия)."""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
warnings.filterwarnings('ignore')

path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/bench_crystallm/full_pub/eval.csv'
LO, HI = 0.917, 1.090
df = pd.read_csv(path)
for c in ('valid', 'comp_ok', 'match_std', 'match_strict', 'sg_ok_0.01', 'sg_ok_0.1'):
    df[c] = df[c].astype(int)
nbp = 'analysis/results/bench_neighbors.csv'
use_nb = os.path.exists(nbp)
if use_nb:
    nb = pd.read_csv(nbp)[['material_id', 'n_same_type']]
    df = df.merge(nb, on='material_id', how='left')
    n0 = df.n_same_type.isna().mean()
    use_nb = n0 < 0.5
    if not use_nb:
        print(f'счётчик соседей есть лишь для {1 - n0:.0%} целей: в модель не включён')
df['E10'] = df.e_above_hull / 0.01
df['log_sg_train'] = np.log1p(df.sg_train_count.clip(lower=0))
df['log_proto_train'] = np.log1p(df.proto_train_count.clip(lower=0))
CTRL = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train']
if use_nb:
    df['log_same_type'] = np.log1p(df.n_same_type)
    CTRL.append('log_same_type')
    df = df[df.log_same_type.notna()]
print(f"файл {path}: попыток {len(df)}, целей {df.material_id.nunique()}, попыток на цель (медиана) {df.groupby('material_id').size().median():.0f}")
print(f"валидных CIF {df.valid.mean():.3f}; состав совпал {df.comp_ok.mean():.3f}")
print('\nдоля по корзинам e_above_hull (все попытки):')
print(df.groupby('bin').agg(N=('k', 'size'), цели=('material_id', 'nunique'), valid=('valid', 'mean'),
      comp_ok=('comp_ok', 'mean'), match_std=('match_std', 'mean'), match_strict=('match_strict', 'mean'),
      sg_ok=('sg_ok_0.1', 'mean')).round(3))
t = df.groupby('material_id').agg(any_std=('match_std', 'max'), any_strict=('match_strict', 'max'), bin=('bin', 'first'), nf=('n_free', 'first'))
print('\nдоля целей, угаданных хотя бы раз за K попыток:')
print(t.groupby('bin')[['any_std', 'any_strict']].mean().round(3))

def fit(d, y, cols):
    z = d.copy()
    for c in CTRL:
        if z[c].std() > 0:
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    cols = [c for c in cols if z[c].std() > 0]
    X = sm.add_constant(z[cols])
    codes = pd.factorize(z.material_id)[0]
    try:
        return sm.GLM(z[y], X, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': codes})
    except Exception as e:
        print('  модель не оценивается:', type(e).__name__)
        return None

def show(m, name='E10'):
    if m is None or name not in m.params:
        return
    b, se = m.params[name], m.bse[name]
    lo90, hi90 = np.exp(b - 1.645 * se), np.exp(b + 1.645 * se)
    ci = m.conf_int().loc[name]
    p_low = 1 - norm.cdf((b - np.log(LO)) / se)   # H0: OR <= LO
    p_high = norm.cdf((b - np.log(HI)) / se)       # H0: OR >= HI
    eq = max(p_low, p_high)
    pdir = norm.cdf(b / se)   # H1: beta<0
    print(f"  ОШ на +10 мэВ = {np.exp(b):.3f}; 95% ДИ [{np.exp(ci[0]):.3f}; {np.exp(ci[1]):.3f}]; 90% ДИ [{lo90:.3f}; {hi90:.3f}]; "
          f"эквивалентность {LO}-{HI}: p={eq:.4f} ({'принята' if eq < 0.05 else 'не принята'}); направленный (β<0) p={pdir:.4f}")

for lab, d in (('n_free >= 1', df[df.n_free >= 1]), ('n_free >= 3', df[df.n_free >= 3])):
    for y in ('match_strict', 'match_std', 'sg_ok_0.1'):
        print(f"\n=== {lab}, {y}: попыток {len(d)}, целей {d.material_id.nunique()} ===")
        show(fit(d, y, ['E10'] + CTRL))
d = df[(df.n_free >= 1) & (df.comp_ok == 1)]
for y in ('match_strict', 'match_std'):
    print(f"\n=== n_free >= 1, только попытки с верным составом, {y}: попыток {len(d)} ===")
    show(fit(d, y, ['E10'] + CTRL))
