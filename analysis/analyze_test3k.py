"""Проверка 3 гипотезы 1 (между составами) на наборе test3k: 3000 целей тестовой части, по 600 в каждой корзине энергии.

Модель (раздел 3 файла гипотез): логистическая регрессия на уровне попыток,
  logit Pr(M=1) = b0 + bE * E/0,01 + контроли (ln(1+n_гр), n_ат, n_своб, ln(1+n_прот), ln(1+n_тип)),
стандартные ошибки кластеризованы по цели (и, для проверки, по составу). Эквивалентность: exp(bE) в границах
[0,917; 1,090] по двум односторонним критериям (90%-й ДИ внутри границ). Дополнительно — ступень «>= 5 мэВ/атом
против < 5» (проверка 10) и разность наклонов двух моделей (взаимодействие энергии с моделью, одни и те же цели).

Входы:
  DiffCSP++ — eval.csv частей outputs_server/bench/test3k_k3_c??/ (eval_bench.py);
  CrystaLLM-small с группой — outputs/bench_crystallm/test3k_sg/eval.csv (eval_crystallm.py);
  ковариаты — analysis/results/bench_test3k_meta.csv, n_тип — analysis/results/bench_neighbors.csv.
У CrystaLLM попытка без CIF в eval.csv не попадает; недостающие до K попытки считаются неудачами.

Использование (окружение bench_stats):
  python analysis/analyze_test3k.py > analysis/results/test3k_between.txt
"""
import argparse
import glob
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--diffcsp', default='outputs_server/bench/test3k_k3_c??/eval.csv')
ap.add_argument('--crystallm', default='outputs/bench_crystallm/test3k_sg/eval.csv')
ap.add_argument('--K', type=int, default=3)
a = ap.parse_args()

R = 'analysis/results'
LO, HI = 0.917, 1.090
meta = pd.read_csv(f'{R}/bench_test3k_meta.csv')
nb = pd.read_csv(f'{R}/bench_neighbors.csv')[['material_id', 'n_same_type', 'capped']].drop_duplicates('material_id')
meta = meta.merge(nb, on='material_id', how='left')
miss = int(meta.n_same_type.isna().sum())
if miss:
    print(f'внимание: у {miss} целей нет n_same_type (нужен nearest_train.py --only-missing); они исключены')
meta = meta.dropna(subset=['n_same_type'])
meta['E10'] = meta.e_above_hull / 0.01
meta['meta'] = (meta.e_above_hull >= 0.005).astype(int)
meta['log_sg_train'] = np.log1p(meta.sg_train_count.clip(lower=0))
meta['log_proto_train'] = np.log1p(meta.proto_train_count.clip(lower=0))
meta['log_same_type'] = np.log1p(meta.n_same_type)
CTRL = ['log_sg_train', 'n_atoms', 'n_free', 'log_proto_train', 'log_same_type']
meta['capped'] = meta.capped.astype(str).str.lower().eq('true')
COV = ['material_id', 'formula', 'bin', 'E10', 'meta', 'capped'] + CTRL


def attempts(files, name):
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in ('material_id', 'match_strict', 'match_std')) for p in files])
    d = d[d.material_id.isin(set(meta.material_id))].copy()
    for c in ('match_strict', 'match_std'):
        d[c] = d[c].fillna(False).astype(bool).astype(int)
    n = d.groupby('material_id').size()
    over = int((n > a.K).sum())
    if over:
        raise SystemExit(f'{name}: у {over} целей больше {a.K} попыток — проверьте входные файлы')
    pad = (a.K - n.reindex(meta.material_id).fillna(0).astype(int))
    pad = pad[pad > 0]
    if len(pad):
        d = pd.concat([d, pd.DataFrame({'material_id': np.repeat(pad.index, pad.values), 'match_strict': 0, 'match_std': 0})])
    d = d.merge(meta[COV], on='material_id', how='inner')
    print(f'{name}: файлов {len(files)}, целей {d.material_id.nunique()}, попыток {len(d)}, '
          f'дописано неудач за отсутствующие попытки {int(pad.sum()) if len(pad) else 0}')
    return d


dc = attempts(sorted(glob.glob(a.diffcsp)), 'DiffCSP++')
cl = attempts(sorted(glob.glob(a.crystallm)), 'CrystaLLM-small')


def fit(d, y, cols, groups='material_id'):
    z = d.copy()
    for c in CTRL:
        if c in cols and z[c].std() > 0:
            z[c] = (z[c] - z[c].mean()) / z[c].std()
    cols = [c for c in cols if z[c].std() > 0]
    X = sm.add_constant(z[cols])
    try:
        return sm.GLM(z[y], X, family=sm.families.Binomial()).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(z[groups])[0]})
    except Exception as e:
        print('  не оценивается:', type(e).__name__)
        return None


def row(m, name, label, eq=True):
    if m is None or name not in m.params:
        return
    b, se = m.params[name], m.bse[name]
    ci, ci90 = m.conf_int().loc[name], m.conf_int(alpha=0.10).loc[name]
    s = (f'  {label:<46} ОШ={np.exp(b):.3f} [{np.exp(ci[0]):.3f}; {np.exp(ci[1]):.3f}]  '
         f'90%: [{np.exp(ci90[0]):.3f}; {np.exp(ci90[1]):.3f}]  p(ОШ=1)={m.pvalues[name]:.4f}')
    if eq:
        peq = max(1 - norm.cdf((b - np.log(LO)) / se), norm.cdf((b - np.log(HI)) / se))
        s += f'  экв. p={peq:.4f} ({"принята" if peq < 0.05 else "не принята"})'
    print(s)


print(f'\nграницы эквивалентности ОШ на +10 мэВ/атом: [{LO}; {HI}]')
for y in ('match_strict', 'match_std'):
    for lab, sel in (('n_free >= 1 (основной отбор проверки 3)', lambda d: d[d.n_free >= 1]), ('все цели', lambda d: d)):
        print(f'\n##### {y}, {lab} #####')
        for name, d0 in (('DiffCSP++, полный шаблон', dc), ('CrystaLLM-small, состав и группа', cl)):
            d = sel(d0)
            print(f'{name}: целей {d.material_id.nunique()}, составов {d.formula.nunique()}, попыток {len(d)}, '
                  f'доля удач {d[y].mean():.3f}')
            print('  доля удач по корзинам:', d.groupby('bin', sort=True)[y].mean().round(3).to_dict())
            row(fit(d, y, ['E10']), 'E10', 'без контролей, +10 мэВ/атом')
            row(fit(d, y, ['E10', 'n_free', 'n_atoms']), 'E10', 'n_free, n_atoms, +10 мэВ/атом')
            row(fit(d, y, ['E10'] + CTRL[:4]), 'E10', 'контроли без n_тип, +10 мэВ/атом')
            m = fit(d, y, ['E10'] + CTRL)
            row(m, 'E10', 'полные контроли, +10 мэВ/атом')
            row(fit(d, y, ['E10'] + CTRL, groups='formula'), 'E10', 'полные контроли, кластеры по составу')
            row(fit(d[~d.capped], y, ['E10'] + CTRL), 'E10', 'полные контроли, без целей с подвыборкой n_тип')
            row(fit(d, y, ['meta'] + CTRL), 'meta', 'ступень >= 5 мэВ/атом (полные контроли)', eq=False)
            m = fit(d, y, ['meta', 'E10'] + CTRL)
            row(m, 'meta', 'ступень при учёте наклона', eq=False)
            row(m, 'E10', 'наклон при учёте ступени, +10 мэВ/атом')
        both = pd.concat([sel(dc).assign(model=1), sel(cl).assign(model=0)], ignore_index=True)
        both['EM'] = both.E10 * both.model
        m = fit(both, y, ['E10', 'model', 'EM'] + CTRL)
        print('разность моделей на одних и тех же целях (DiffCSP++ против CrystaLLM-small, полные контроли):')
        row(m, 'model', 'DiffCSP++ при E=0, шансы', eq=False)
        row(m, 'EM', 'отношение ОШ наклонов (DiffCSP++ / CrystaLLM)', eq=False)
