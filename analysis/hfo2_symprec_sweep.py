"""symprec-sweep по сэмплам HfO2 (H1): куда уходят «промахи» при разных допусках."""
import sys, json
from collections import Counter
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

run_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
SYMPRECS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.05, 0.1, 0.2, 0.3]
TARGETS = {'hfo2_stable': 14, 'hfo2_meta': 29}

def sg(s, sp):
    try:
        return SpacegroupAnalyzer(s, symprec=sp).get_space_group_number()
    except Exception:
        return -1

result = {}
for label, tgt in TARGETS.items():
    cifs = sorted((run_dir / label).glob('*.cif'), key=lambda p: (len(p.stem), p.stem))
    structs = [Structure.from_file(str(p)) for p in cifs]
    table = {sp: [sg(s, sp) for s in structs] for sp in SYMPRECS}
    result[label] = {'files': [p.name for p in cifs], 'target': tgt,
                     'sg_by_symprec': {str(k): v for k, v in table.items()}}
    print(f"\n=== {label} (цель SG{tgt}, N={len(structs)}) ===")
    for sp in SYMPRECS:
        c = Counter(table[sp])
        dist = ', '.join(f'SG{k}:{v}' for k, v in c.most_common())
        print(f"symprec={sp:<8g} попаданий {c.get(tgt, 0):>2}/{len(structs)}  {dist}")

REF = 0.1
for label, tgt in TARGETS.items():
    d = result[label]
    miss = [i for i, v in enumerate(d['sg_by_symprec'][str(REF)]) if v != tgt]
    if miss:
        print(f"\n--- {label}: не попали в SG{tgt} при symprec={REF}; SG по допускам ---")
        print('file   ' + ' '.join(f'{sp:>7g}' for sp in SYMPRECS))
        for i in miss:
            row = ' '.join(f"{d['sg_by_symprec'][str(sp)][i]:>7}" for sp in SYMPRECS)
            print(f"{d['files'][i]:<7}{row}")

out_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(result, open(out_json, 'w'), indent=1)
