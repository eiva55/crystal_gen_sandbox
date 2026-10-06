"""StructureMatcher (match rate, RMSE) сэмплов HfO2 к эталонам mp-352 и mp-685097."""
import sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

run_dir, refs_dir = Path(sys.argv[1]), Path(sys.argv[2])
load = lambda p: Structure.from_dict(json.load(open(p)))
REFS = {'stable': load(refs_dir / 'mp-352.json'), 'meta': load(refs_dir / 'mp-685097.json')}
MATCHERS = {'default': StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5),
            'loose': StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)}

allrows = {}
for label in ['hfo2_stable', 'hfo2_meta']:
    cifs = sorted((run_dir / label).glob('*.cif'), key=lambda p: (len(p.stem), p.stem))
    rows = []
    for p in cifs:
        s = Structure.from_file(str(p))
        row = {'file': p.name, 'sg01': SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()}
        for mn, m in MATCHERS.items():
            for rn, r in REFS.items():
                row[f'fit_{mn}_{rn}'] = bool(m.fit(s, r))
                rms = m.get_rms_dist(s, r)
                row[f'rms_{mn}_{rn}'] = None if rms is None else float(rms[0])
        rows.append(row)
    allrows[label] = rows

    print(f"\n=== {label} (N={len(rows)}) ===")
    for mn in MATCHERS:
        for rn in REFS:
            k = f'fit_{mn}_{rn}'
            rms = [r[f'rms_{mn}_{rn}'] for r in rows if r[f'rms_{mn}_{rn}'] is not None]
            print(f"  {mn:<8} к эталону {rn:<6}: match {sum(r[k] for r in rows):>2}/{len(rows)}, "
                  f"средний RMS (по совпавшим) {np.mean(rms) if rms else float('nan'):.4f}")
    by_sg = defaultdict(list)
    for r in rows:
        by_sg[r['sg01']].append(r)
    print("  по группе при symprec=0.1 (match default → stable / meta):")
    for g, rr in sorted(by_sg.items()):
        print(f"    SG{g}: n={len(rr)}, "
              f"stable {sum(x['fit_default_stable'] for x in rr)}, "
              f"meta {sum(x['fit_default_meta'] for x in rr)}")

json.dump(allrows, open('analysis/results/hfo2_matcher.json', 'w'), indent=1)
