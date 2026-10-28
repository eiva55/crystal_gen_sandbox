"""Зависит ли убывание доли удач с энергией внутри состава от того, есть ли у состава полиморф в обучающей части MP-20.

Разведочная проверка возможной причины эффекта внутри состава: модель могла выучить другой полиморф того же состава,
и метастабильная цель «проигрывает» ему. Цели наборов пар лежат в test и val, поэтому сами они в обучении не встречаются;
признак относится к составу:
  G_train   — в обучающей части есть хотя бы одна структура того же приведённого состава;
  G_ground  — самый низкий по энергии полиморф состава во всей MP-20 лежит в обучающей части;
  ln(1 + n_train) — число структур того же состава в обучающей части (непрерывный вариант).
Признак постоянен внутри состава, поэтому сам он поглощается фиксированным эффектом, а оценивается взаимодействие:
  доля_d = b · E10_d + c · (G · E10)_d + контроли_d,
наклон при G = 0 — b, при G = 1 — b + c. Модель, кластеры и отборы — те же, что в within_strata.py.

Использование (окружение bench_stats):
  python analysis/train_polymorph_interaction.py --label diffcsp_tv_k10 > analysis/results/train_polymorph_diffcsp_tv_k10.txt
  python analysis/train_polymorph_interaction.py --label sg_tv_k10
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pymatgen.core import Composition
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--data', default='models/diffcsp_pp/data/mp_20')
ap.add_argument('--results', default='analysis/results')
a = ap.parse_args()
R = a.results


def red(f):
    try:
        return Composition(f).reduced_formula
    except Exception:
        return f


ps = pd.read_csv(f'{R}/pairs_{a.label}_per_structure.csv')
pp = pd.read_csv(f'{R}/pairs_{a.label}_per_pair.csv')
ps = ps.merge(pd.read_csv(f'{R}/bench_neighbors.csv')[['material_id', 'n_same_type']].drop_duplicates('material_id'),
              on='material_id', how='left')
mp = pd.concat([pd.read_csv(f'{a.data}/{s}.csv', usecols=['material_id', 'pretty_formula', 'e_above_hull']).assign(split=s)
                for s in ('train', 'val', 'test')])
mp['rf'] = mp.pretty_formula.map(red)
n_train = mp[mp.split == 'train'].groupby('rf').size()
ground = mp.sort_values('e_above_hull').drop_duplicates('rf').set_index('rf').split
ps['rf'] = ps.formula.map(red)
ps['n_train'] = ps.rf.map(n_train).fillna(0).astype(int)
ps['G_train'] = (ps.n_train > 0).astype(int)
ps['G_ground'] = (ps.rf.map(ground) == 'train').astype(int)
ps['L_train'] = np.log1p(ps.n_train)
ps['E10'] = ps.e_above_hull / 0.01
ps['log_sgc'] = np.log1p(ps.sg_train_count.clip(lower=0))
ps['log_pc'] = np.log1p(ps.proto_train_count.clip(lower=0))
ps['log_nst'] = np.log1p(ps.n_same_type)
CTRL = ['log_sgc', 'n_atoms', 'n_free', 'log_pc', 'log_nst']

pp['dn'] = pp.nf_m - pp.nf_s
both = (pp.nf_s >= 1) & (pp.nf_m >= 1)
SUBSETS = [('все пары', pp.dn.notna()),
           ('основной отбор: |Δn| <= 2, оба n_free >= 1', (pp.dn.abs() <= 2) & both)]


def fit(z, extra, ctrl):
    z = z.copy()
    for c in extra:
        z[c + 'xE'] = z[c] * z.E10
    cols = ['E10'] + [c + 'xE' for c in extra] + ctrl
    for c in ['rate'] + cols:
        z[c + '_d'] = z[c] - z.groupby('formula')[c].transform('mean')
    X = sm.add_constant(z[[c + '_d' for c in cols]])
    return sm.OLS(z['rate_d'], X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(z.formula)[0]})


def show(m, name, lab):
    b = m.params[name]; lo, hi = m.conf_int().loc[name]
    print(f'    {lab:42s} {b:+.4f}  95% ДИ [{lo:+.4f}; {hi:+.4f}]  p={m.pvalues[name]:.3f}')


print(f'метка {a.label}; изменение доли удачных попыток на +10 мэВ/атом внутри состава; кластеры — составы')
for lab, sel in SUBSETS:
    q = pp[sel]
    z = ps[ps.material_id.isin(set(q.id_s) | set(q.id_m))]
    z = z[z.groupby('formula').material_id.transform('size') >= 2].copy()
    comp = z.drop_duplicates('formula')
    print(f'\n== {lab}: пар {len(q)}, структур {len(z)}, составов {z.formula.nunique()}')
    print(f'   составов с полиморфом в обучении: {int(comp.G_train.sum())} из {len(comp)}; '
          f'с нижним полиморфом в обучении: {int(comp.G_ground.sum())}; '
          f'медиана числа структур состава в обучении {comp.n_train.median():.0f}')
    for g in ('G_train', 'G_ground'):
        for v in (0, 1):
            zz = z[z[g] == v]
            if zz.formula.nunique() >= 10:
                m = fit(zz, [], CTRL)
                show(m, 'E10_d', f'{g} = {v} ({zz.formula.nunique()} составов), наклон')
            else:
                print(f'    {g} = {v}: составов {zz.formula.nunique()}, меньше 10 — не оценивается')
        if z[g].nunique() == 2:
            m = fit(z, [g], CTRL)
            show(m, f'{g}xE_d', f'{g}: разность наклонов (1 − 0)')
    m = fit(z, ['L_train'], CTRL)
    show(m, 'E10_d', 'ln(1 + n_train) = 0: наклон')
    show(m, 'L_trainxE_d', 'изменение наклона на +1 ln(1 + n_train)')
