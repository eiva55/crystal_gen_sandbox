"""Стабильный против метастабильного полиморфа одного состава: сколько попыток из K удачны для каждой цели.

Входы: один или несколько eval.csv (eval_crystallm.py, eval_published.py или eval_bench.py) по целям набора;
попытки из разных файлов объединяются, номер попытки k перенумеровывается. Необязательно — результат cross_match.py.

Выходы (analysis/results/):
  pairs_<label>_per_structure.csv — по каждой цели: попыток, валидных, верный состав, верная группа, совпадений
  pairs_<label>_per_pair.csv      — по каждой паре «стабильный — метастабильный»: k_s из K, k_m из K, разность энергий
и сводка в stdout (средняя разность долей, «хотя бы одна удача», знаковый критерий, модель внутри состава).

Использование:
  python analysis/analyze_pairs_v1.py --set pairs_v1 --label pub --eval outputs/bench_crystallm/full_pub/eval.csv
  python analysis/analyze_pairs_v1.py --set pairs_v1 --label sg --eval outputs/bench_crystallm/pair_sg/eval.csv outputs/bench_crystallm/pairs_v1_sg/eval.csv
  python analysis/analyze_pairs_v1.py --set pairs_v1nf --label diffcsp --eval "outputs/bench/pv1_o*/eval.csv"
"""
import argparse
import glob
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--set', default='pairs_v1')
ap.add_argument('--label', required=True)
ap.add_argument('--eval', nargs='+', required=True)
ap.add_argument('--cross', default='', help='CSV из cross_match.py (необязательно)')
ap.add_argument('--outcome', default='match_strict', choices=['match_strict', 'match_std'])
a = ap.parse_args()
rng = np.random.default_rng(0)

meta = pd.read_csv(f'analysis/results/bench_{a.set}_meta.csv')
ids = set(meta.material_id)
frames, base = [], 0
for pattern in a.eval:
    for p in sorted(glob.glob(pattern)):
        d = pd.read_csv(p)
        d = d[d.material_id.isin(ids)].copy()
        if d.empty:
            continue
        d['k'] = d.groupby('material_id').cumcount() + base     # попытки разных файлов не пересекаются
        base = int(d.k.max()) + 1
        d['source'] = p
        frames.append(d)
if not frames:
    raise SystemExit('в eval.csv нет целей этого набора')
ev = pd.concat(frames)
if 'valid' not in ev:                    # eval_bench.py (DiffCSP++): CIF есть — значит валиден
    ev['valid'] = ev.generated
if 'comp_ok' not in ev:
    ev['comp_ok'] = ev.aligned
for c in ('valid', 'comp_ok', 'match_std', 'match_strict', 'sg_ok_0.1'):
    ev[c] = ev[c].fillna(False).astype(bool).astype(int)

ps = ev.groupby('material_id').agg(K=('k', 'size'), valid=('valid', 'sum'), comp_ok=('comp_ok', 'sum'),
                                   sg_ok=('sg_ok_0.1', 'sum'), strict=('match_strict', 'sum'),
                                   std=('match_std', 'sum')).reset_index()
ps = meta.merge(ps, on='material_id', how='inner')
if a.cross:
    cr = pd.read_csv(a.cross)
    cr = cr[cr.material_id.isin(ids)]
    tab = pd.crosstab(cr.material_id, cr.category)
    tab.columns = ['n_' + c.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') for c in tab.columns]
    ps = ps.merge(tab.reset_index(), on='material_id', how='left')
y = 'strict' if a.outcome == 'match_strict' else 'std'
ps['rate'] = ps[y] / ps.K
ps.to_csv(f'analysis/results/pairs_{a.label}_per_structure.csv', index=False)

rows = []
for f, g in ps.groupby('formula'):
    s, ms = g[g.pair_role == 'stable'], g[g.pair_role == 'meta']
    if s.empty or ms.empty:
        continue
    s = s.iloc[0]
    for _, m in ms.iterrows():
        rows.append({'formula': f, 'id_s': s.material_id, 'id_m': m.material_id, 'sg_s': s.sg, 'sg_m': m.sg,
                     'nat_s': s.n_atoms, 'nat_m': m.n_atoms, 'nf_s': s.n_free, 'nf_m': m.n_free,
                     'E_s': s.e_above_hull, 'E_m': m.e_above_hull, 'dE_meV': (m.e_above_hull - s.e_above_hull) * 1000,
                     'K_s': s.K, 'K_m': m.K, 'k_s': s[y], 'k_m': m[y], 'p_s': s.rate, 'p_m': m.rate})
r = pd.DataFrame(rows)
if r.empty:
    raise SystemExit('нет пар со стабильным и метастабильным полиморфом')
r['category'] = np.select([(r.k_s > 0) & (r.k_m > 0), (r.k_s > 0) & (r.k_m == 0), (r.k_s == 0) & (r.k_m > 0)],
                          ['оба', 'только стабильный', 'только метастабильный'], 'ни один')
r.to_csv(f'analysis/results/pairs_{a.label}_per_pair.csv', index=False)


def cluster_ci(d, col, B=3000):
    """95% ДИ среднего бутстрэпом по составам (пары одного состава зависимы)."""
    groups = [g[col].values for _, g in d.groupby('formula')]
    bs = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


print(f"набор {a.set}, метка {a.label}, исход {a.outcome}; целей {len(ps)}, попыток на цель: "
      f"медиана {ps.K.median():.0f} (от {ps.K.min()} до {ps.K.max()})")
print(f"доля целей без единой удачи: стабильные {(ps[ps.pair_role == 'stable'][y] == 0).mean():.2f}, "
      f"метастабильные {(ps[ps.pair_role == 'meta'][y] == 0).mean():.2f}")
subsets = [('все пары', r), ('оба n_free >= 1', r[(r.nf_s >= 1) & (r.nf_m >= 1)]),
           ('одинаковое число атомов', r[r.nat_s == r.nat_m])]
for lab, d in subsets:
    if len(d) < 5:
        continue
    d = d.copy()
    d['dp'] = d.p_s - d.p_m
    d['dany'] = (d.k_s > 0).astype(int) - (d.k_m > 0).astype(int)
    pc = d.groupby('formula').dp.mean()
    kpos, nz = int((pc > 0).sum()), int((pc != 0).sum())
    pv = binomtest(kpos, nz).pvalue if nz else float('nan')
    lo, hi = cluster_ci(d, 'dp')
    lo2, hi2 = cluster_ci(d, 'dany')
    cats = d.category.value_counts().to_dict()
    print(f"\n{lab}: пар {len(d)}, составов {d.formula.nunique()}, медиана разности энергий {d.dE_meV.median():.1f} мэВ/атом")
    print(f"  средняя доля удачных попыток: стабильный {d.p_s.mean():.3f}, метастабильный {d.p_m.mean():.3f}; "
          f"разность {d.dp.mean():+.3f} [{lo:+.3f}; {hi:+.3f}]")
    print(f"  хотя бы одна удача: стабильный {(d.k_s > 0).mean():.2f}, метастабильный {(d.k_m > 0).mean():.2f}; "
          f"разность {d.dany.mean():+.3f} [{lo2:+.3f}; {hi2:+.3f}]")
    print(f"  по составам: стабильный лучше в {kpos} из {nz} составов с различием, знаковый критерий p={pv:.3f}")
    print(f"  категории пар: {cats}")

# модель внутри состава: доля удачных попыток на +10 мэВ/атом при фиксированном составе (линейная, кластеры — составы)
z = ps[ps.groupby('formula').material_id.transform('size') >= 2].copy()
z['E10'] = z.e_above_hull / 0.01
z['log_sgc'] = np.log1p(z.sg_train_count.clip(lower=0))
z['log_pc'] = np.log1p(z.proto_train_count.clip(lower=0))
variants = [('без контролей', []), ('с n_free, n_atoms', ['n_free', 'n_atoms'])]
try:
    nb = pd.read_csv('analysis/results/bench_neighbors.csv')[['material_id', 'n_same_type']]
    z = z.merge(nb, on='material_id', how='left')
    z['log_nst'] = np.log1p(z.n_same_type)
    if z.log_nst.notna().all():
        variants.append(('полные контроли (n_гр, n_ат, n_своб, n_прот, n_тип)', ['log_sgc', 'n_atoms', 'n_free', 'log_pc', 'log_nst']))
except FileNotFoundError:
    pass
for lab, ctrl in variants:
    zz = z.copy()
    for c in ['E10', 'rate'] + ctrl:
        zz[c + '_d'] = zz[c] - zz.groupby('formula')[c].transform('mean')
    X = sm.add_constant(zz[['E10_d'] + [c + '_d' for c in ctrl]])
    m = sm.OLS(zz['rate_d'], X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(zz.formula)[0]})
    ci95, ci90 = m.conf_int().loc['E10_d'], m.conf_int(alpha=0.10).loc['E10_d']
    print(f"\nвнутри состава ({lab}): {m.params['E10_d']:+.4f} на +10 мэВ/атом, 95% ДИ [{ci95[0]:+.4f}; {ci95[1]:+.4f}], "
          f"90% ДИ [{ci90[0]:+.4f}; {ci90[1]:+.4f}] (граница эквивалентности ±0,0216)")
if a.cross:
    cr = cr.merge(meta[['material_id', 'pair_role']], on='material_id', how='left', suffixes=('', '_m'))
    print('\nчто выдала модель (доли попыток):')
    print(pd.crosstab(cr.pair_role, cr.category, normalize='index').round(3).to_string())
