"""BVS-отклонение у РЕЛАКСИРОВАННЫХ эталонов MP (замечание 2.7): физика антисайта."""
import numpy as np
from mp_api.client import MPRester
from pymatgen.analysis.bond_valence import calculate_bv_sum

key = None
for line in open('.env'):
    if line.startswith('MP_API_KEY='):
        key = line.split('=', 1)[1].strip().strip('"')

FV = {'Mg': 2.0, 'Al': 3.0, 'O': -2.0}

def devs(structure, cutoff=2.6):
    s = structure.copy()
    s.add_oxidation_state_by_element(FV)
    out = {'all': [], 'regular': [], 'antistructural': []}
    for site in s:
        el = site.specie.symbol
        if el not in ('Mg', 'Al'):
            continue
        nb = s.get_neighbors(site, r=cutoff)
        co = len([n for n in nb if n.specie.symbol == 'O'])
        try:
            d = abs(calculate_bv_sum(site, nb) - FV[el])
        except Exception:
            continue
        out['all'].append(d)
        regular = (el == 'Mg' and co == 4) or (el == 'Al' and co == 6)
        out['regular' if regular else 'antistructural'].append(d)
    return out

with MPRester(key) as mpr:
    for mid, name in [('mp-3536', 'x=0 (эталон MP)'), ('mp-34330', 'x=0.5 (эталон MP)')]:
        d = devs(mpr.get_structure_by_material_id(mid))
        print(f"\n{mid} {name}: D(s)={np.mean(d['all']):.4f}")
        for k in ('regular', 'antistructural'):
            if d[k]:
                print(f"   {k}: n={len(d[k])}, среднее {np.mean(d[k]):.4f}")

print("\nДля сравнения, сгенерированные (пилот): x=0 D=0.1205, x=0.5 D=0.2378; "
      "штатные 0.1452, антиструктурные 0.5270")
