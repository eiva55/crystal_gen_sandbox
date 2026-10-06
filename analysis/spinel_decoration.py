"""Какую раскладку катионов реально выдаёт модель: координация Mg и Al в примитивной ячейке."""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import calculate_bv_sum

FV = {'Mg': 2.0, 'Al': 3.0, 'O': -2.0}
run_dir = Path(sys.argv[1])

def analyse(s, cutoff=2.6):
    s = s.get_primitive_structure()
    n = len(s)
    s = s.copy()
    s.add_oxidation_state_by_element(FV)
    mg, al, devs = [], [], []
    for site in s:
        el = site.specie.symbol
        if el not in ('Mg', 'Al'):
            continue
        nb = s.get_neighbors(site, r=cutoff)
        cn = sum(1 for x in nb if x.specie.symbol == 'O')
        (mg if el == 'Mg' else al).append(cn)
        try:
            devs.append(abs(calculate_bv_sum(site, nb) - FV[el]))
        except Exception:
            pass
    return n, tuple(sorted(mg)), tuple(sorted(al)), float(np.mean(devs))

load = lambda p: Structure.from_dict(json.load(open(p)))
for name in ['mp-3536', 'mp-34330']:
    n, mg, al, d = analyse(load(f'analysis/refs/{name}.json'))
    print(f"эталон {name}: атомов {n}, Mg CN {mg}, Al CN {al}, D={d:.3f}")

for label in ['spinel_x0_ordered', 'spinel_x05_disordered']:
    pats = Counter()
    print(f"\n=== {label} ===")
    for p in sorted((run_dir / label).glob('*.cif'), key=lambda p: (len(p.stem), p.stem)):
        n, mg, al, d = analyse(Structure.from_file(str(p)))
        pats[(mg, al)] += 1
        print(f"{p.name:<8} атомов {n:>2}  Mg CN {mg}  Al CN {al}  D={d:.3f}")
    print("паттерны (Mg CN, Al CN): число структур")
    for k, v in pats.most_common():
        print("  ", k, v)
