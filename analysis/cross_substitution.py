"""Замещение при промахе: какой полиморф выдаёт модель вместо цели.

Для каждой цели считаем долю её попыток, закончившихся строгим совпадением с ДРУГИМ известным полиморфом того же состава:
  - со стабильным (энергия над оболочкой < 5 мэВ/атом),
  - с метастабильным (энергия >= 5 мэВ/атом).
Сравниваем две величины, как в гипотезе: метастабильная цель замещается стабильным полиморфом против стабильная цель
замещается другим полиморфом (любым / метастабильным). Усреднение по составам, 95% ДИ — бутстрэп по составам (B=3000).
Запуск: python analysis/cross_substitution.py analysis/results/cross_pairs_v1_sg_k10.csv
"""
import sys
import numpy as np
import pandas as pd

E_STAB = 0.005
df = pd.read_csv(sys.argv[1])
df['miss'] = ~df.category.isin(['цель'])
strict = df.category == 'другой полиморф'
loose = df.category.isin(['другой полиморф', 'другой полиморф (станд.)'])
for nm, m in (('строго', strict), ('строго+станд.', loose)):
    df[f'{nm}|стаб'] = (m & (df.matched_E < E_STAB)).astype(int)
    df[f'{nm}|мета'] = (m & (df.matched_E >= E_STAB)).astype(int)
    df[f'{nm}|любой'] = m.astype(int)

g = df.groupby(['formula', 'material_id', 'pair_role']).agg(
    n=('k', 'size'), miss=('miss', 'sum'),
    **{c: (c, 'sum') for c in df.columns if '|' in c}).reset_index()
cols = [c for c in g.columns if '|' in c]
comp = g.groupby(['formula', 'pair_role'])[['n', 'miss'] + cols].sum().unstack('pair_role')
comp = comp.dropna()                                   # только составы, где есть обе роли
print(f'целей {g.shape[0]} (стабильных {(g.pair_role=="stable").sum()}, метастабильных {(g.pair_role=="meta").sum()}); '
      f'составов с обеими ролями {len(comp)}; попыток на цель {int(g.n.median())}')

rng = np.random.default_rng(0)
B = 3000
idx = rng.integers(0, len(comp), (B, len(comp)))


def est(c_m, c_s, den, sel):
    """Доли по пулу попыток (den='n') или по промахам (den='miss') для выбранных составов."""
    num_m, num_s = comp[(c_m, 'meta')].values[sel].sum(-1), comp[(c_s, 'stable')].values[sel].sum(-1)
    den_m, den_s = comp[(den, 'meta')].values[sel].sum(-1), comp[(den, 'stable')].values[sel].sum(-1)
    return num_m / den_m, num_s / den_s


P = []
ROW_PREFIX = ''


def row(label, cm, cs, den):
    a, b = est(cm, cs, den, np.arange(len(comp)))
    am, bm = est(cm, cs, den, idx)
    d = am - bm
    lo, hi = np.percentile(d, [2.5, 97.5])
    P.append((f'{ROW_PREFIX}{label}', a - b, 2 * min(max((d <= 0).mean(), 1 / B), max((d >= 0).mean(), 1 / B))))
    print(f'{label:62s} метаст.: {a:.3f}  стаб.: {b:.3f}  разность: {a-b:+.3f} [{lo:+.3f}; {hi:+.3f}]')


for nm in ('строго', 'строго+станд.'):
    for den, dn in (('n', 'от всех попыток'), ('miss', 'от промахов')):
        print(f'\n== {nm}, {dn} ==')
        ROW_PREFIX = f'{nm}, {dn}: '
        row('метастабильная цель -> стабильный полиморф | стабильная цель -> метастабильный',
            f'{nm}|стаб', f'{nm}|мета', den)
        row('метастабильная цель -> стабильный полиморф | стабильная цель -> любой другой',
            f'{nm}|стаб', f'{nm}|любой', den)
        row('метастабильная цель -> любой другой полиморф | стабильная цель -> любой другой',
            f'{nm}|любой', f'{nm}|любой', den)


# поправка Holm на все сравнения (двусторонний бутстрэп-p; грубая оценка, нулевое значение не центрировалось)
order = np.argsort([p for _, _, p in P])
m = len(P)
adj = np.zeros(m)
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * P[i][2]))
    adj[i] = run
print(f'\n== поправка Holm, сравнений {m} ==')
for (lab, dlt, p), pa in zip(P, adj):
    print(f'{lab[:100]:100s} разность {dlt:+.3f}  p={p:.3f}  p(Holm)={pa:.3f}')
