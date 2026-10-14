"""Оценка генераций CrystaLLM (после postprocess): совпадение с эталоном, группа, валидность; формат eval.csv как у DiffCSP++."""
import glob
import os
import sys
import tarfile
import warnings
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings('ignore')

tag = sys.argv[1]                       # например v0_sg
setname = tag.rsplit('_', 1)[0]
out = f'outputs/bench_crystallm/{tag}'
meta = pd.read_csv(f'analysis/results/bench_{setname}_meta.csv').set_index('material_id')
D = 'models/diffcsp_pp/data/mp_20'
splits = sorted(set(meta['split'])) if 'split' in meta else ['test']
ref_cif = pd.concat([pd.read_csv(f'{D}/{s}.csv', usecols=['material_id', 'cif']) for s in splits]).set_index('material_id')['cif']
M = {'std': StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10),
     'strict': StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)}
refs, rows = {}, []
for post in sorted(glob.glob(f'{out}/post_*.tar.gz')):
    with tarfile.open(post, 'r:gz') as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            mid, k = os.path.basename(m.name)[:-4].rsplit('__', 1)
            if mid not in meta.index:
                continue
            text = tar.extractfile(m).read().decode()
            row = {'material_id': mid, 'k': int(k) - 1, **meta.loc[mid].to_dict(), 'generated': True, 'aligned': True,
                   'valid': False, 'comp_ok': False, 'match_std': False, 'match_strict': False,
                   'rmse_std': None, 'rmse_strict': None, 'sg_ok_0.01': False, 'sg_ok_0.1': False}
            try:
                gen = CifParser.from_str(text).parse_structures(primitive=True)[0]
                row['valid'] = True
            except Exception:
                rows.append(row)
                continue
            if mid not in refs:
                refs[mid] = CifParser.from_str(ref_cif.loc[mid]).parse_structures(primitive=True)[0]
            ref = refs[mid]
            row['comp_ok'] = gen.composition.reduced_formula == ref.composition.reduced_formula
            if row['comp_ok']:
                for n, mm in M.items():
                    r = mm.get_rms_dist(gen, ref)
                    row[f'match_{n}'] = r is not None
                    row[f'rmse_{n}'] = None if r is None else r[0]
            for sp in (0.01, 0.1):
                try:
                    g = SpacegroupAnalyzer(gen, symprec=sp).get_space_group_number()
                except Exception:
                    g = -1
                row[f'sg_ok_{sp}'] = (g == row['sg'])
            rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f'{out}/eval.csv', index=False)
print(f"{tag}: генераций {len(df)}, целей {df.material_id.nunique()} из {len(meta)}; "
      f"валидных CIF {df.valid.mean():.2f}, состав совпал {df.comp_ok.mean():.2f}")
print(df.groupby('bin', sort=False)[['valid', 'comp_ok', 'match_std', 'match_strict', 'sg_ok_0.01', 'sg_ok_0.1']].mean().round(2))
