"""Окно допуска эталонов для всех структур test MP-20: поиск «хрупких» (псевдосимметричных) и полярных целей."""
import sys
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
warnings.filterwarnings('ignore')

GRID = np.logspace(-3, np.log10(0.5), 25)
POLAR = set([1] + list(range(3, 10)) + list(range(25, 47)) + list(range(75, 81)) + list(range(99, 111)) +
            list(range(143, 147)) + list(range(156, 162)) + list(range(168, 174)) + list(range(183, 187)))

def work(args):
    mid, cif, tgt = args
    warnings.filterwarnings('ignore')
    try:
        s = CifParser.from_str(cif).parse_structures(primitive=True)[0]
    except Exception:
        return mid, np.nan, 0
    last = np.nan
    for sp in GRID:
        try:
            g = SpacegroupAnalyzer(s, symprec=float(sp)).get_space_group_number()
        except Exception:
            g = -1
        if g != tgt:
            break
        last = float(sp)
    return mid, last, len(s)

if __name__ == '__main__':
    test = pd.read_csv('models/diffcsp_pp/data/mp_20/test.csv',
                       usecols=['material_id', 'pretty_formula', 'e_above_hull', 'spacegroup.number', 'cif'])
    jobs = list(zip(test.material_id, test.cif, test['spacegroup.number'].astype(int)))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        res = p.map(work, jobs, chunksize=20)
    r = pd.DataFrame(res, columns=['material_id', 'tau_ref', 'n_atoms'])
    out = test.drop(columns='cif').merge(r, on='material_id').rename(columns={'spacegroup.number': 'sg', 'pretty_formula': 'formula'})
    out['polar'] = out.sg.isin(POLAR)
    out.to_csv('analysis/results/test_tau_ref.csv', index=False)
    v = out[out.tau_ref.notna()]
    print(f"всего {len(out)}, с определённым окном {len(v)}; полярных {int(v.polar.sum())}")
    for t in (0.05, 0.1, 0.2, 0.3):
        print(f"окно эталона < {t} Å: всего {int((v.tau_ref < t).sum())}, полярных {int(((v.tau_ref < t) & v.polar).sum())}")
    print("окно эталона >= 0.49 Å (держит группу до конца сетки):", int((v.tau_ref >= 0.49).sum()))
    f = v[(v.tau_ref < 0.3) & (v.n_atoms <= 20)]
    print("\nхрупкие цели (окно < 0.3 Å) по корзинам e_above_hull:")
    print(f.groupby(pd.cut(f.e_above_hull, [-1, 0.005, 0.02, 0.04, 0.06, 0.081]), observed=True).agg(
        N=('tau_ref', 'size'), polar=('polar', 'sum')))
