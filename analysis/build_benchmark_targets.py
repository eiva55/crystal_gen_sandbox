"""Стратифицированный набор целей из test MP-20 (невиденные структуры) для CSP-бенчмарка."""
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

D = 'models/diffcsp_pp/data/mp_20'
PER_BIN, SEED = 60, 0
BINS = [(0, 0.005), (0.005, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.0801)]

test = pd.read_csv(f'{D}/test.csv')
train_sg = pd.read_csv(f'{D}/train.csv', usecols=['spacegroup.number'])['spacegroup.number'].value_counts()
rng = np.random.default_rng(SEED)

targets, meta = [], []
for lo, hi in BINS:
    pool = test[(test.e_above_hull >= lo) & (test.e_above_hull < hi)]
    pool = pool.sample(min(len(pool), PER_BIN * 2), random_state=SEED)  # запас на отбраковку
    kept = 0
    for _, row in pool.iterrows():
        if kept >= PER_BIN:
            break
        try:
            s = CifParser.from_str(row['cif']).parse_structures(primitive=True)[0]
            ds = SpacegroupAnalyzer(s, symprec=0.1).get_symmetry_dataset()
        except Exception:
            continue
        if int(ds['number']) != int(row['spacegroup.number']) or len(s) > 20:
            continue
        species = [site.specie.symbol for site in s]
        groups = defaultdict(list)
        for i, eq in enumerate(ds['equivalent_atoms']):
            groups[eq].append(i)
        types, letters = [], []
        for idx in groups.values():
            types.append(species[idx[0]])
            letters.append(f"{len(idx)}{ds['wyckoffs'][idx[0]]}")
        targets.append({'label': row['material_id'], 'spacegroup_number': int(ds['number']),
                        'wyckoff_letters': letters, 'atom_types': types})
        meta.append({'material_id': row['material_id'], 'formula': row['pretty_formula'],
                     'e_above_hull': row['e_above_hull'], 'bin': f'{lo}-{hi}',
                     'sg': int(ds['number']), 'n_atoms': len(s), 'n_orbits': len(groups),
                     'sg_train_count': int(train_sg.get(int(ds['number']), 0))})
        kept += 1
    print(f"корзина {lo}-{hi}: отобрано {kept}")

json.dump(targets, open('configs/csp_targets/bench_v0.json', 'w'), indent=1)
pd.DataFrame(meta).to_csv('analysis/results/bench_v0_meta.csv', index=False)
print(f"всего целей: {len(targets)}")
print(pd.DataFrame(meta).groupby('bin')[['n_atoms', 'n_orbits', 'sg_train_count']].median())
