"""Эффект энергии с поправкой на покрытие (n_same_type): (1) весь test, опубликованные генерации; (2) 300 целей v0: с группой против только состава;
(3) ступень «почти стабильные против остальных» и наклон внутри метастабильных. Стандартные ошибки кластеризованы по цели."""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
warnings.filterwarnings('ignore')

tag = sys.argv[1] if len(sys.argv) > 1 else 'v0'
LO, HI = 0.917, 1.090
nb = pd.read_csv('analysis/results/bench_neighbors.csv')[['material_id', 'n_same_type', 'capped']]
pub = pd.read_csv('outputs/bench_crystallm/full_pub/eval.csv').merge(nb, on='material_id', how='inner')
sg = pd.read_csv(f'outputs/bench_crystallm/{tag}_sg/eval.csv').merge(nb, on='material_id', how='inner')
for d in (pub, sg):
    d['E10'] = d.e_above_hull / 0.01
    d['meta'] = (d.e_above_hull >= 0.005).astype(int)
    d['log_sg_train'] = np.log1p(d.sg_train_count.clip(lower=0))
    d['log_proto_train'] = np.log1p(d.proto_train_count.clip(lower=0))
    d['log_same_type'] = np.log1p(d.n_same_type)
    for c in ('match_strict', 'match_std', 'sg_ok_0.1', 'comp_ok'):
        d[c] = d[c].astype(int)
CTRL = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train', 'log_same_type']

def fit(d, y, cols, weights=None):
    z = d.copy()
    for c in CTRL:
        if c in cols and z[c].std() > 0:
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    cols = [c for c in cols if z[c].std() > 0]
    X = sm.add_constant(z[cols])
    codes = pd.factorize(z.material_id)[0]
    try:
        kw = {} if weights is None else {'var_weights': weights}
        return sm.GLM(z[y], X, family=sm.families.Binomial(), **kw).fit(cov_type='cluster', cov_kwds={'groups': codes}), len(z)
    except Exception as e:
        print('  не оценивается:', type(e).__name__)
        return None, len(z)

def row(m, name, label, per=''):
    if m is None or name not in m.params:
        return
    b, se = m.params[name], m.bse[name]
    if not np.isfinite(se) or se > 5:
        print(f"  {label:<44} не оценивается (разделение или вырождение)")
        return
    ci = m.conf_int().loc[name]
    eq = max(1 - norm.cdf((b - np.log(LO)) / se), norm.cdf((b - np.log(HI)) / se))
    print(f"  {label:<44} ОШ{per}={np.exp(b):.3f} [{np.exp(ci[0]):.3f}; {np.exp(ci[1]):.3f}]  p(ОШ=1)={m.pvalues[name]:.4f}  экв. p={eq:.4f}")

print(f"соседи посчитаны для целей: pub {pub.material_id.nunique()}, {tag}_sg {sg.material_id.nunique()}")
for y in ('match_strict', 'match_std'):
    print(f"\n##### 1. Весь test, только состав, K=20: {y} #####")
    d = pub[pub.n_free >= 1]
    print(f"цели n_free>=1: {d.material_id.nunique()}, попыток {len(d)}")
    m, _ = fit(d, y, ['E10'] + CTRL[:4]); row(m, 'E10', 'без покрытия, линейно по E, +10 мэВ', '')
    m, _ = fit(d, y, ['E10'] + CTRL); row(m, 'E10', 'с покрытием, линейно по E, +10 мэВ', '')
    m, _ = fit(d[d.capped == 0], y, ['E10'] + CTRL); row(m, 'E10', 'с покрытием, без capped-целей, +10 мэВ', '')
    m, _ = fit(d, y, ['meta'] + CTRL); row(m, 'meta', 'ступень: >=5 мэВ против <5 мэВ (с покрытием)', ' ')
    m, _ = fit(d[d.meta == 1], y, ['E10'] + CTRL); row(m, 'E10', 'наклон только среди >=5 мэВ, +10 мэВ', '')
    m, _ = fit(d, y, ['meta', 'E10'] + CTRL)
    row(m, 'meta', 'ступень при учёте наклона', ' '); row(m, 'E10', 'наклон при учёте ступени, +10 мэВ', '')

print(f"\n##### 2. Те же {tag}-цели: с группой (K=1) против только состав (K=20) #####")
ids = set(sg.material_id)
p = pub[pub.material_id.isin(ids) & (pub.n_free >= 1)]
s = sg[sg.n_free >= 1]
for y in ('match_strict', 'match_std'):
    print(f"\n{y}: цели n_free>=1: {s.material_id.nunique()}")
    m, _ = fit(s, y, ['E10'] + CTRL); row(m, 'E10', 'с группой, +10 мэВ', '')
    m, _ = fit(p, y, ['E10'] + CTRL); row(m, 'E10', 'только состав, +10 мэВ', '')
    g = p.groupby('material_id').agg(rate=(y, 'mean'), n=(y, 'size'), **{c: (c, 'first') for c in ['E10', 'meta'] + CTRL}).reset_index()
    g = g.rename(columns={'rate': y}); g['grp'] = 0
    s2 = s[['material_id', y, 'E10', 'meta'] + CTRL].copy(); s2['n'] = 1; s2['grp'] = 1
    both = pd.concat([g, s2], ignore_index=True)
    both['EG'] = both.E10 * both.grp
    m, _ = fit(both, y, ['E10', 'grp', 'EG'] + CTRL, weights=both.n.values)
    row(m, 'grp', 'группа в запросе (при E=0), шансы', ' ')
    row(m, 'EG', 'разница наклонов (с группой − без), +10 мэВ', '')
