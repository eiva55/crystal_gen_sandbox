"""Те же цели набора: CrystaLLM с группой в запросе (K=1) против опубликованных генераций только по составу (K=20)."""
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

tag = sys.argv[1] if len(sys.argv) > 1 else 'v0'
sg = pd.read_csv(f'outputs/bench_crystallm/{tag}_sg/eval.csv')
pub = pd.read_csv('outputs/bench_crystallm/full_pub/eval.csv')
pub = pub[pub.material_id.isin(sg.material_id)]
Y = ['comp_ok', 'match_strict', 'match_std', 'sg_ok_0.1']
for d in (sg, pub):
    for c in Y:
        d[c] = d[c].astype(int)
a = sg.groupby('bin')[Y].mean().add_suffix('_sg')
b = pub.groupby('bin')[Y].mean().add_suffix('_pub')
print(f"целей {sg.material_id.nunique()}; попыток: с группой {len(sg)}, только состав {len(pub)}")
print(a.join(b)[[f'{y}_{s}' for y in Y for s in ('sg', 'pub')]].round(2).to_string())

for name, d in (('с группой', sg), ('только состав', pub)):
    z = d[d.comp_ok == 1]
    print(f"\n{name}, только попытки с верным составом (N={len(z)}): match_strict по корзинам")
    print(z.groupby('bin').match_strict.agg(['mean', 'size']).round(2).T.to_string())

t = sg.set_index('material_id')[['match_strict', 'e_above_hull', 'bin']].join(
    pub.groupby('material_id').match_strict.mean().rename('pub_rate'))
t['diff'] = t.match_strict - t.pub_rate
print(f"\nсредняя разность (с группой − только состав) по целям: {t['diff'].mean():+.3f}; по корзинам:")
print(t.groupby('bin')['diff'].agg(['mean', 'size']).round(3).T.to_string())
se = t['diff'].std() / np.sqrt(len(t))
print(f"общая: {t['diff'].mean():+.3f} ± {1.96 * se:.3f} (95%)")

for name, d in (('с группой', sg), ('только состав', pub)):
    z = d.copy(); z['E10'] = z.e_above_hull / 0.01
    m = sm.GLM(z.match_strict, sm.add_constant(z[['E10']]), family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(z.material_id)[0]})
    ci = m.conf_int().loc['E10']
    print(f"ОШ на +10 мэВ (без ковариат), {name}: {np.exp(m.params['E10']):.3f} [{np.exp(ci[0]):.3f}; {np.exp(ci[1]):.3f}]")
