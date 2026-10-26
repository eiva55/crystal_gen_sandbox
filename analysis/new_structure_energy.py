"""Энергия «новых структур» (верный состав, не совпала ни с одним известным полиморфом) по оценке CHGNet.

Для каждой такой попытки считается энергия на атом генерации и сравнивается с энергией цели (та же модель, те же шаги)
и со стабильным полиморфом того же состава. Окружение с chgnet: orchestrator.
Запуск:
  python analysis/new_structure_energy.py --cross analysis/results/cross_pairs_v1_sg_k10.csv \
    --tar "outputs/bench_crystallm/pair_sg/post_*.tar.gz" "outputs/bench_crystallm/pairs_v1_sg/post_*.tar.gz" \
    --out analysis/results/new_structure_energy_k10.csv [--relax-steps 0] [--limit 20]
Попытка задаётся тройкой (source, material_id, k) из cross_match.py: при объединении прогонов номера k повторяются.
Файл сопоставления без столбца source принимается, только если пары (цель, k) в нём не повторяются.
--relax-steps 0: одно вычисление без релаксации (быстро, но сгенерированные структуры не релаксированы, а цели взяты
из релаксированной базы, поэтому ΔE смещена вверх); --relax-steps 60: короткая релаксация всех трёх структур.
"""
import argparse, glob, os, tarfile, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('--cross', required=True)
ap.add_argument('--tar', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--relax-steps', type=int, default=0)
ap.add_argument('--limit', type=int, default=0, help='только первые N попыток (проверка)')
ap.add_argument('--data', default='models/diffcsp_pp/data/mp_20')
a = ap.parse_args()
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

cr = pd.read_csv(a.cross)
if 'source' not in cr:
    if cr.duplicated(['material_id', 'k']).any():
        raise SystemExit('в файле сопоставления повторяются (цель, k) и нет столбца source: перезапусти cross_match.py')
    cr['source'] = ''
new = cr[cr.category == 'новая структура'][['source', 'material_id', 'k', 'formula', 'pair_role', 'e_above_hull']].copy()
if a.limit:
    new = new.head(a.limit)
want = set(zip(new.source, new.material_id, new.k))
targets = set(new.material_id)
# стабильный полиморф состава = цель роли stable того же состава
stab = cr[cr.pair_role == 'stable'].drop_duplicates('formula').set_index('formula').material_id.to_dict()
need_ids = targets | {stab[f] for f in new.formula if f in stab}
cols = ['material_id', 'cif']
mp = pd.concat([pd.read_csv(f'{a.data}/{s}.csv', usecols=cols) for s in ('train', 'val', 'test')])
mp = mp[mp.material_id.isin(need_ids)].drop_duplicates('material_id').set_index('material_id').cif.to_dict()


def split_name(name):
    b = os.path.basename(name); b = b[:-4] if b.endswith('.cif') else b
    mid, k = b.rsplit('__', 1) if '__' in b else b.rsplit('_', 1)
    return mid, int(k)


gens = {}
for pat in a.tar:
    for path in sorted(glob.glob(pat)):
        with tarfile.open(path, 'r:*') as tar:
            for m in tar:
                if m.isfile() and m.name.endswith('.cif'):
                    try:
                        key = split_name(m.name)
                    except ValueError:
                        continue
                    key = (path if (path, *key) in want else '', *key)
                    if key in want:
                        gens[key] = tar.extractfile(m).read().decode()
print(f'нужно попыток {len(want)}, найдено CIF {len(gens)}', flush=True)
if len(gens) < len(want):
    print('внимание: часть CIF не найдена — пути --tar должны совпадать с теми, что были у cross_match.py', flush=True)

from chgnet.model import CHGNet, StructOptimizer
model = CHGNet.load()
opt = StructOptimizer(model=model) if a.relax_steps > 0 else None


def energy(s):
    try:
        if opt is not None:
            r = opt.relax(s, steps=a.relax_steps, verbose=False)
            return float(r['trajectory'].energies[-1] / len(s))
        return float(model.predict_structure(s)['e'])
    except Exception:
        return np.nan


cache = {}


def ref_energy(mid):
    if mid not in cache:
        cache[mid] = energy(CifParser.from_str(mp[mid]).parse_structures(primitive=True)[0]) if mid in mp else np.nan
    return cache[mid]


rows = []
for i, r in enumerate(new.itertuples(), 1):
    txt = gens.get((r.source, r.material_id, r.k))
    if txt is None:
        continue
    try:
        s = CifParser.from_str(txt).parse_structures(primitive=True)[0]
    except Exception:
        continue
    e_gen = energy(s)
    e_t = ref_energy(r.material_id)
    e_s = ref_energy(stab.get(r.formula, ''))
    rows.append(dict(source=r.source, material_id=r.material_id, k=r.k, formula=r.formula, pair_role=r.pair_role, n_atoms=len(s),
                     e_gen=e_gen, e_target=e_t, e_stable=e_s, d_target=e_gen - e_t, d_stable=e_gen - e_s))
    if i % 50 == 0:
        print(f'{i}/{len(new)}', flush=True)
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
d = df.dropna(subset=['d_target'])
print(f'\nпопыток с энергией: {len(d)} из {len(new)}; релаксация: {a.relax_steps} шагов')
for role, g in d.groupby('pair_role'):
    comp = g.groupby('formula').apply(lambda x: pd.Series({'ниже цели': (x.d_target < 0).mean(),
                                                           'ниже стабильного': (x.d_stable < 0).mean()}))
    print(f'{role:7s} попыток {len(g):4d}  составов {g.formula.nunique():3d}  медиана E_ген−E_цели {g.d_target.median()*1000:+7.1f} мэВ/атом  '
          f'медиана E_ген−E_стаб. {g.d_stable.median()*1000:+7.1f}  '
          f'доля ниже цели (по составам) {comp["ниже цели"].mean():.3f}  ниже стабильного {comp["ниже стабильного"].mean():.3f}')
