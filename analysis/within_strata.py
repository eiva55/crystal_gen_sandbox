"""Модель внутри состава по отборам пар: все пары, оба полиморфа с n_free >= 1, страты по сложности шаблона.

Вход — файлы analyze_pairs_v1.py: pairs_<label>_per_structure.csv и pairs_<label>_per_pair.csv; ковариата n_same_type
из bench_neighbors.csv. Модель та же, что в analyze_pairs_v1.py: доля удачных попыток на +10 мэВ/атом, фиксированный
эффект состава (центрирование внутри состава), кластеры — составы. В отбор попадают структуры из выбранных пар; составы
с одной структурой отбрасываются. Δn = n_free(метастабильный) − n_free(стабильный).

--exclude-degenerate: убрать пары, в которых хотя бы у одного полиморфа шаблон общий с другим полиморфом того же состава
(analysis/results/template_degeneracy.csv): для таких целей условие не различает полиморфы.

Использование (окружение bench_stats):
  python analysis/within_strata.py --label sg_k10 > analysis/results/within_strata_sg_k10.txt
  python analysis/within_strata.py --label diffcsp_tv_k10 --exclude-degenerate analysis/results/template_degeneracy.csv
  python analysis/within_strata.py --label diffcsp_tv_k10 --only-set pairs_val_v1nf   # отдельно часть val
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--delta', type=float, default=0.0216, help='граница эквивалентности, доли на +10 мэВ/атом')
ap.add_argument('--exclude-degenerate', default='', help='template_degeneracy.csv: исключить пары с неоднозначным шаблоном')
ap.add_argument('--only-set', default='', help='оставить только цели набора (например pairs_v1nf или pairs_val_v1nf): разбор по частям MP-20')
a = ap.parse_args()

R = 'analysis/results'
ps = pd.read_csv(f'{R}/pairs_{a.label}_per_structure.csv')
pp = pd.read_csv(f'{R}/pairs_{a.label}_per_pair.csv')
ps = ps.merge(pd.read_csv(f'{R}/bench_neighbors.csv')[['material_id', 'n_same_type']], on='material_id', how='left')
if a.only_set:
    keep = set(pd.read_csv(f'{R}/bench_{a.only_set}_meta.csv').material_id)
    ps = ps[ps.material_id.isin(keep)]
    pp = pp[pp.id_s.isin(keep) & pp.id_m.isin(keep)]
    print(f'только цели набора {a.only_set}: структур {len(ps)}, пар {len(pp)}')
ps['E10'] = ps.e_above_hull / 0.01
ps['log_sgc'] = np.log1p(ps.sg_train_count.clip(lower=0))
ps['log_pc'] = np.log1p(ps.proto_train_count.clip(lower=0))
ps['log_nst'] = np.log1p(ps.n_same_type)
if a.exclude_degenerate:
    deg = set(' '.join(pd.read_csv(a.exclude_degenerate).members).split())
    n0 = len(pp)
    pp = pp[~pp.id_s.isin(deg) & ~pp.id_m.isin(deg)]
    print(f'исключены пары с неоднозначным шаблоном: {n0 - len(pp)} из {n0}')
pp['dn'] = pp.nf_m - pp.nf_s
both = (pp.nf_s >= 1) & (pp.nf_m >= 1)

VARIANTS = [('без контролей', []), ('n_free, n_atoms', ['n_free', 'n_atoms']),
            ('полные контроли', ['log_sgc', 'n_atoms', 'n_free', 'log_pc', 'log_nst'])]
SUBSETS = [('все пары', pp.dn.notna()),
           ('все пары, оба n_free >= 1', both),
           ('сопоставимые |Δn| <= 2', pp.dn.abs() <= 2),
           ('сопоставимые |Δn| <= 2, оба n_free >= 1 (основной критерий проверки 6)', (pp.dn.abs() <= 2) & both),
           ('Δn = 0', pp.dn == 0),
           ('Δn = 0, оба n_free >= 1', (pp.dn == 0) & both),
           ('хотя бы у одного n_free = 0', ~both)]


def fit(z, ctrl):
    z = z.copy()
    for c in ['E10', 'rate'] + ctrl:
        z[c + '_d'] = z[c] - z.groupby('formula')[c].transform('mean')
    X = sm.add_constant(z[['E10_d'] + [c + '_d' for c in ctrl]])
    m = sm.OLS(z['rate_d'], X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(z.formula)[0]})
    return m.params['E10_d'], m.conf_int().loc['E10_d'].values, m.conf_int(alpha=0.10).loc['E10_d'].values


print(f'метка {a.label}; изменение доли удачных попыток на +10 мэВ/атом внутри состава; '
      f'эквивалентность — 90%-й ДИ внутри ±{a.delta}')
for lab, sel in SUBSETS:
    q = pp[sel]
    z = ps[ps.material_id.isin(set(q.id_s) | set(q.id_m))]
    z = z[z.groupby('formula').material_id.transform('size') >= 2]
    print(f'\n{lab}: пар {len(q)}, структур {len(z)}, составов {z.formula.nunique()}; '
          f'средняя разность p_s − p_m {(q.p_s - q.p_m).mean():+.3f}')
    if z.formula.nunique() < 10:
        print('  составов меньше 10, модель не оценивается')
        continue
    for vl, ctrl in VARIANTS:
        b, c95, c90 = fit(z, ctrl)
        eq = 'принята' if (c90[0] > -a.delta and c90[1] < a.delta) else 'не принята'
        print(f'  {vl:16s} {b:+.4f}  95% ДИ [{c95[0]:+.4f}; {c95[1]:+.4f}]  90% ДИ [{c90[0]:+.4f}; {c90[1]:+.4f}]  '
              f'эквивалентность {eq}')
