"""Оценка готовых (опубликованных) генераций CrystaLLM по целям набора: совпадение с эталоном, состав, группа, валидность.
Использование: python eval_published.py <набор> <архив.tar.gz> [K] [NPROC]. Результат: outputs/bench_crystallm/<набор>_pub/eval.csv"""
import os
import sys
import tarfile
import warnings
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings('ignore')

setname, archive = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NPROC = int(sys.argv[4]) if len(sys.argv) > 4 else 6
D = 'models/diffcsp_pp/data/mp_20'
meta = pd.read_csv(f'analysis/results/bench_{setname}_meta.csv').set_index('material_id')
splits = sorted(set(meta['split'])) if 'split' in meta else ['test']
REF = pd.concat([pd.read_csv(f'{D}/{s}.csv', usecols=['material_id', 'cif']) for s in splits]).set_index('material_id')['cif'].to_dict()
SG = meta['sg'].to_dict()
MS = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
MT = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

def split_name(path):
    b = os.path.basename(path)
    b = b[:-4] if b.endswith('.cif') else b
    if '__' in b:
        mid, k = b.rsplit('__', 1)
    else:
        mid, k = b.rsplit('_', 1)
    return mid, int(k)

@lru_cache(maxsize=256)
def ref_of(mid):
    return CifParser.from_str(REF[mid]).parse_structures(primitive=True)[0]

def work(item):
    mid, k, text = item
    warnings.filterwarnings('ignore')
    row = {'material_id': mid, 'k': k, 'generated': True, 'aligned': True, 'valid': False, 'comp_ok': False,
           'match_std': False, 'match_strict': False, 'rmse_std': None, 'rmse_strict': None,
           'sg_ok_0.01': False, 'sg_ok_0.1': False}
    try:
        gen = CifParser.from_str(text).parse_structures(primitive=True)[0]
        row['valid'] = True
        ref = ref_of(mid)
        row['comp_ok'] = gen.composition.reduced_formula == ref.composition.reduced_formula
        if row['comp_ok']:
            r = MS.get_rms_dist(gen, ref)
            row['match_std'] = r is not None
            row['rmse_std'] = None if r is None else r[0]
            if r is not None:          # строгие допуски тоньше стандартных: проверяем только при стандартном совпадении
                r2 = MT.get_rms_dist(gen, ref)
                row['match_strict'] = r2 is not None
                row['rmse_strict'] = None if r2 is None else r2[0]
        for sp in (0.01, 0.1):
            try:
                gg = SpacegroupAnalyzer(gen, symprec=sp).get_space_group_number()
            except Exception:
                gg = -1
            row[f'sg_ok_{sp}'] = (gg == SG[mid])
    except Exception:
        pass
    return row

def items():
    seen, skipped = Counter(), Counter()
    with tarfile.open(archive, 'r:*') as tar:
        for m in tar:
            if not m.isfile():
                continue
            try:
                mid, k = split_name(m.name)
            except Exception:
                skipped['имя'] += 1
                continue
            if mid not in meta.index:
                skipped['нет в наборе'] += 1
                continue
            if seen[mid] >= K:
                skipped['сверх K'] += 1
                continue
            seen[mid] += 1
            yield mid, k, tar.extractfile(m).read().decode(errors='replace')
    print('пропущено файлов:', dict(skipped), flush=True)

if __name__ == '__main__':
    out = f'outputs/bench_crystallm/{setname}_pub'
    os.makedirs(out, exist_ok=True)
    rows = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(work, items(), chunksize=40)):
            rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f'обработано {i + 1}', flush=True)
    df = pd.DataFrame(rows).merge(meta.reset_index(), on='material_id', how='left')
    df.to_csv(f'{out}/eval.csv', index=False)
    print(f"{setname}_pub: генераций {len(df)}, целей {df.material_id.nunique()} из {len(meta)}; "
          f"валидных {df.valid.mean():.3f}, состав совпал {df.comp_ok.mean():.3f}")
    print(df.groupby('bin')[['valid', 'comp_ok', 'match_std', 'match_strict', 'sg_ok_0.01', 'sg_ok_0.1']].mean().round(3))
