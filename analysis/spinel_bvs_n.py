"""Шпинель MgAl₂O₄: отклонение суммы валентных связей D по раскладкам на больших выборках (проверки 1 и 2 гипотезы 4).

Для каждого образца: раскладка по числу ближайших атомов кислорода (радиус 2,6 Å; как в spinel_decoration.py) и
D = среднее |BVS − v| по катионам (метод Брауна–Альтермата, pymatgen, формальные валентности Mg²⁺, Al³⁺, O²⁻).
  Проверка 1: D упорядоченных образцов на шаблоне R3m против D упорядоченных образцов на шаблоне Fd-3m
              (H1: в R3m больше; критерий Манна–Уитни): понижение симметрии само по себе не ухудшает локальную химию.
  Проверка 2: ΔD = D − D(релаксированный эталон той же раскладки); эталоны mp-3536 (упорядоченная) и mp-34330
              (антисайтная) — значения D из analysis/results (spinel_ref_bvs.py): 0,052 и 0,284; H1: ΔD > 0
              (знаковый критерий Уилкоксона).
Образцы с другими паттернами (нефизичные) исключаются и перечисляются.

Использование (окружение mp_lookup):
  python analysis/spinel_bvs_n.py outputs_server/spinel_n > analysis/results/spinel_bvs_server.txt
"""
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import calculate_bv_sum
warnings.filterwarnings('ignore')

FV = {'Mg': 2.0, 'Al': 3.0, 'O': -2.0}
REF_D = {'упорядоченная': 0.052, 'антисайтная': 0.284}
root = Path(sys.argv[1])


def analyse(s, cutoff=2.6):
    s = s.copy()
    s.add_oxidation_state_by_element(FV)
    cn = {'Mg': [], 'Al': []}
    dev = []
    for site in s:
        el = site.specie.symbol
        if el not in cn:
            continue
        nb = s.get_neighbors(site, r=cutoff)
        cn[el].append(sum(1 for n in nb if n.specie.symbol == 'O'))
        try:
            dev.append(abs(calculate_bv_sum(site, nb) - FV[el]))
        except Exception:
            pass
    mg, al = sorted(cn['Mg']), sorted(cn['Al'])
    k = len(mg) // 2                       # число примитивных ячеек по 14 атомов
    if mg == [4] * len(mg) and al == [6] * len(al):
        arr = 'упорядоченная'
    elif mg == [4] * k + [6] * k and al == [4] * k + [6] * (len(al) - k):
        arr = 'антисайтная'
    else:
        arr = 'прочая'
    return arr, (float(np.mean(dev)) if dev else np.nan)


res = {}
for label, tpl in (('spinel_x05_disordered', 'R3m'), ('spinel_x0_ordered', 'Fd-3m')):
    for p in sorted((root / label).glob('*.cif'), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
        arr, d = analyse(Structure.from_file(str(p)))
        res.setdefault((tpl, arr), []).append((p.name, d))

print('шаблон, раскладка: образцов, среднее D, медиана D')
for (tpl, arr), v in sorted(res.items()):
    d = np.array([x[1] for x in v])
    print(f'  {tpl:6s} {arr:14s} {len(v):4d}  {np.nanmean(d):.3f}  {np.nanmedian(d):.3f}' + ('' if arr != 'прочая' else f'  файлы: {[x[0] for x in v]}'))

D = lambda tpl, arr: np.array([x[1] for x in res.get((tpl, arr), [])], dtype=float)
a, b = D('R3m', 'упорядоченная'), D('Fd-3m', 'упорядоченная')
if len(a) and len(b):
    u = mannwhitneyu(a, b, alternative='greater')
    print(f'\nПроверка 1: D упорядоченных в R3m ({len(a)}, медиана {np.median(a):.3f}) против Fd-3m ({len(b)}, медиана '
          f'{np.median(b):.3f}): U={u.statistic:.0f}, p={u.pvalue:.4f} (H1: в R3m больше)')
print('\nПроверка 2: ΔD относительно релаксированного эталона той же раскладки (H1: ΔD > 0)')
for tpl, arr in (('R3m', 'антисайтная'), ('R3m', 'упорядоченная'), ('Fd-3m', 'упорядоченная')):
    x = D(tpl, arr) - REF_D[arr]
    x = x[~np.isnan(x)]
    if len(x) >= 5:
        w = wilcoxon(x, alternative='greater')
        print(f'  {tpl:6s} {arr:14s} образцов {len(x):4d}  медиана ΔD {np.median(x):+.3f}  доля ΔD > 0 {np.mean(x > 0):.2f}  p={w.pvalue:.4g}')
