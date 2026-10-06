"""Релаксация CHGNet: эталоны и сэмплы. Сохраняется ли группа/энергия после релаксации."""
import sys, json, time
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from chgnet.model import StructOptimizer

hfo2_dir, spinel_dir = Path(sys.argv[1]), Path(sys.argv[2])
srt = lambda d: sorted(d.glob('*.cif'), key=lambda p: (len(p.stem), p.stem))

jobs = [(f'ref_{n}', Structure.from_dict(json.load(open(f'analysis/refs/{n}.json'))))
        for n in ['mp-352', 'mp-685097', 'mp-3536', 'mp-34330']]
for label, k in [('hfo2_meta', 20), ('hfo2_stable', 10)]:
    jobs += [(f'{label}/{p.name}', Structure.from_file(str(p))) for p in srt(hfo2_dir / label)[:k]]
for label in ['spinel_x0_ordered', 'spinel_x05_disordered']:
    jobs += [(f'{label}/{p.name}', Structure.from_file(str(p))) for p in srt(spinel_dir / label)]

def sg(s, sp):
    try:
        return SpacegroupAnalyzer(s, symprec=sp).get_space_group_number()
    except Exception:
        return -1

relaxer = StructOptimizer(use_device='cpu')
out = []
for name, s in jobs:
    t0 = time.time()
    s = s.get_primitive_structure()
    r = relaxer.relax(s, fmax=0.05, steps=500, verbose=False)
    fin, en = r['final_structure'], r['trajectory'].energies
    rec = {'name': name, 'n_atoms': len(s), 'steps': len(en),
           'E0': float(en[0]) / len(s), 'E1': float(en[-1]) / len(s),
           'sg_before': {str(sp): sg(s, sp) for sp in (0.01, 0.1)},
           'sg_after': {str(sp): sg(fin, sp) for sp in (0.01, 0.1)},
           'sec': round(time.time() - t0, 1)}
    out.append(rec)
    print(rec, flush=True)
    json.dump(out, open('analysis/results/relax_chgnet.json', 'w'), indent=1)
