"""Скачивает эталонные структуры MP в JSON (CIF в .gitignore)."""
import json
from pathlib import Path
from mp_api.client import MPRester

key = None
for line in open('.env'):
    if line.startswith('MP_API_KEY='):
        key = line.split('=', 1)[1].strip().strip('"')

IDS = ['mp-352', 'mp-685097', 'mp-3536', 'mp-34330']
out = Path('analysis/refs')
out.mkdir(parents=True, exist_ok=True)
with MPRester(key) as mpr:
    for mid in IDS:
        s = mpr.get_structure_by_material_id(mid)
        json.dump(s.as_dict(), open(out / f'{mid}.json', 'w'))
        print(mid, s.composition.reduced_formula, len(s))
