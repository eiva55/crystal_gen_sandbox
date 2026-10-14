"""Внутри одного состава: зависит ли успех генерации от энергии структуры (цели с одинаковой формулой, разные полиморфы).
Состав фиксирован фиксированным эффектом: сравниваются полиморфы одного состава друг с другом.
Использование: python analyze_pairs.py [eval.csv]   (по умолчанию опубликованные генерации CrystaLLM на всём test)"""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest
warnings.filterwarnings('ignore')

path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/bench_crystallm/full_pub/eval.csv'
rng = np.random.default_rng(0)
df = pd.read_csv(path)
nb = pd.read_csv('analysis/results/bench_neighbors.csv')[['material_id', 'n_same_type']]
df = df.merge(nb, on='material_id', how='left')
for c in ('match_strict', 'match_std', 'comp_ok', 'sg_ok_0.1'):
    df[c] = df[c].astype(int)
agg = df.groupby('material_id').agg(formula=('formula', 'first'), E=('e_above_hull', 'first'), n_atoms=('n_atoms', 'first'),
                                    n_free=('n_free', 'first'), sgc=('sg_train_count', 'first'), pc=('proto_train_count', 'first'),
                                    nst=('n_same_type', 'first'), n=('k', 'size'), comp_ok=('comp_ok', 'mean'),
                                    match_strict=('match_strict', 'mean'), match_std=('match_std', 'mean'),
                                    sg_ok=('sg_ok_0.1', 'mean')).reset_index()
agg['E10'] = agg.E / 0.01
agg['log_sgc'] = np.log1p(agg.sgc.clip(lower=0))
agg['log_pc'] = np.log1p(agg.pc.clip(lower=0))
agg['log_nst'] = np.log1p(agg.nst)
cnt = agg.groupby('formula').material_id.transform('size')
a = agg[cnt >= 2].copy()
print(f"{path}\nсоставов с >=2 целями: {a.formula.nunique()}, целей {len(a)}, попыток на цель (медиана) {a.n.median():.0f}")
print(f"разброс энергии внутри состава (макс - мин), мэВ: медиана {a.groupby('formula').E.agg(lambda x: x.max() - x.min()).median() * 1000:.1f}")

def demean(d, cols):
    z = d.copy()
    for c in cols:
        z[c + '_d'] = z[c] - z.groupby('formula')[c].transform('mean')
    return z

def fe(d, y, extra):
    z = demean(d.assign(), ['E10', y] + extra)
    z = z[z.groupby('formula').material_id.transform('size') >= 2]
    X = sm.add_constant(z[['E10_d'] + [c + '_d' for c in extra]])
    m = sm.OLS(z[y + '_d'], X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(z.formula)[0]})
    b, ci = m.params['E10_d'], m.conf_int().loc['E10_d']
    return b, ci[0], ci[1], m.pvalues['E10_d']

print("\n=== Внутри состава (фиксированные эффекты состава), линейная вероятностная модель ===")
print("изменение доли успеха (в долях единицы) на +10 мэВ/атом; ДИ 95%, кластеры = составы")
for y in ('match_strict', 'match_std', 'sg_ok', 'comp_ok'):
    for lab, extra in (('без ковариат', []), ('с log_sg, n_atoms, n_free, log_proto, log_same_type', ['log_sgc', 'n_atoms', 'n_free', 'log_pc', 'log_nst'])):
        ex = [c for c in extra if a.dropna(subset=extra)[c].std() > 0] if extra else []
        d = a.dropna(subset=ex) if ex else a
        b, lo, hi, p = fe(d, y, ex)
        print(f"  {y:<13} {lab:<52} {b:+.4f} [{lo:+.4f}; {hi:+.4f}] p={p:.4f}")

print("\n=== Парное сравнение: почти стабильный (<5 мэВ) против метастабильного (>=5 мэВ) полиморфа одного состава ===")
rows = []
for f, g in a.groupby('formula'):
    s, m = g[g.E < 0.005], g[g.E >= 0.005]
    if len(s) and len(m):
        rows.append({'formula': f, 'ns': len(s), 'nm': len(m), 'ds': s.match_strict.mean() - m.match_strict.mean(),
                     'dstd': s.match_std.mean() - m.match_std.mean(), 'dsg': s.sg_ok.mean() - m.sg_ok.mean(),
                     'dE': (m.E.mean() - s.E.mean()) * 1000})
r = pd.DataFrame(rows)
if r.empty:
    raise SystemExit('нет составов, где есть и стабильный, и метастабильный полиморф')
print(f"составов с обоими типами: {len(r)}; средняя разность энергий, мэВ: {r.dE.mean():.1f}")
if len(r) >= 5:
    for col, lab in (('ds', 'match_strict'), ('dstd', 'match_std'), ('dsg', 'sg_ok')):
        boots = [r[col].values[rng.integers(0, len(r), len(r))].mean() for _ in range(3000)]
        k = int((r[col] > 0).sum()); nz = int((r[col] != 0).sum())
        pv = binomtest(k, nz, 0.5).pvalue if nz else float('nan')
        print(f"  {lab:<13} стабильный минус метастабильный: {r[col].mean():+.3f} [{np.percentile(boots, 2.5):+.3f}; {np.percentile(boots, 97.5):+.3f}]; "
              f"стабильный лучше в {k} из {nz} составов с различием (знаковый критерий p={pv:.4f})")
