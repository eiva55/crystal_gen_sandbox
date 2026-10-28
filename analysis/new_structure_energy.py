"""Энергия «новых структур» (верный состав, не совпала ни с одним известным полиморфом) по оценке CHGNet.

Для каждой такой попытки считается энергия на атом генерации и сравнивается с энергией цели (та же модель, те же шаги)
и со стабильным полиморфом того же состава. Окружение с chgnet: orchestrator.
Запуск:
  python analysis/new_structure_energy.py --cross analysis/results/cross_pairs_v1_sg_k10.csv \
    --tar "outputs/bench_crystallm/pair_sg/post_*.tar.gz" "outputs/bench_crystallm/pairs_v1_sg/post_*.tar.gz" \
    --out analysis/results/new_structure_energy_k10.csv [--relax-steps 0] [--limit 20]
  DiffCSP++ (папки прогонов run_bench.py вместо архивов; --hits — ещё и выборка удачных попыток):
  python analysis/new_structure_energy.py --cross analysis/results/cross_pairs_tv_nf_diffcsp_k10.csv \
    --bench-dir "outputs_server/bench/pairs_v1nf_k10_c??" "outputs_server/bench/pairs_val_v1nf_k10_c??" \
    --relax-steps 150 --hits 300 --out analysis/results/new_structure_energy_diffcsp_tv_k10.csv
Попытка задаётся тройкой (source, material_id, k) из cross_match.py: при объединении прогонов номера k повторяются.
Файл сопоставления без столбца source принимается, только если пары (цель, k) в нём не повторяются.
--hits N: случайная выборка N строгих совпадений с целью (поровну стабильных и метастабильных целей, seed 0): для них
после той же релаксации записываются группа до и после (допуски 0,1 и 0,01 Å), энергия образца и цели и сохраняется ли
строгое совпадение с целью; результат — <out>_hits.csv.
--relax-steps 0: одно вычисление без релаксации (быстро, но сгенерированные структуры не релаксированы, а цели взяты
из релаксированной базы, поэтому ΔE смещена вверх); --relax-steps 60: короткая релаксация всех трёх структур.
"""
import argparse, glob, os, tarfile, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('--cross', required=True)
ap.add_argument('--tar', nargs='*', default=[])
ap.add_argument('--bench-dir', nargs='*', default=[], help='папки прогонов DiffCSP++ (order.json + <номер>.cif), как у cross_match.py')
ap.add_argument('--hits', type=int, default=0)
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
hits = cr[cr.category == 'цель'][['source', 'material_id', 'k', 'formula', 'pair_role', 'e_above_hull']]
if a.hits:
    hits = pd.concat([g.sample(min(len(g), a.hits // 2), random_state=0) for _, g in hits.groupby('pair_role')])
else:
    hits = hits.head(0)
want = set(zip(new.source, new.material_id, new.k)) | set(zip(hits.source, hits.material_id, hits.k))
targets = set(new.material_id) | set(hits.material_id)
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
import json
for pat in a.bench_dir:
    for d in sorted(glob.glob(pat)):
        order = json.load(open(os.path.join(d, 'order.json')))
        cifs = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(d, '*.cif')) if os.path.basename(p)[:-4].isdigit()}
        if not cifs:
            continue
        off = min(cifs)
        for i, o in enumerate(order):
            key = (d, o['material_id'], o['k'])
            if key in want and (i + off) in cifs:
                gens[key] = open(cifs[i + off]).read()
print(f'нужно попыток {len(want)}, найдено CIF {len(gens)}', flush=True)
if len(gens) < len(want):
    print('внимание: часть CIF не найдена — пути --tar должны совпадать с теми, что были у cross_match.py', flush=True)

from chgnet.model import CHGNet, StructOptimizer
model = CHGNet.load()
opt = StructOptimizer(model=model) if a.relax_steps > 0 else None


def energy(s, keep=False):
    try:
        if opt is not None:
            r = opt.relax(s, steps=a.relax_steps, verbose=False)
            e = float(r['trajectory'].energies[-1] / len(s))
            return (e, r['final_structure']) if keep else e
        e = float(model.predict_structure(s)['e'])
        return (e, s) if keep else e
    except Exception:
        return (np.nan, None) if keep else np.nan


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

if len(hits):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    SM = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    def sgn(x, p):
        try:
            return SpacegroupAnalyzer(x, symprec=p).get_space_group_number()
        except Exception:
            return -1
    hrows = []
    for i, r in enumerate(hits.itertuples(), 1):
        txt = gens.get((r.source, r.material_id, r.k))
        if txt is None:
            continue
        try:
            s = CifParser.from_str(txt).parse_structures(primitive=True)[0]
        except Exception:
            continue
        ref = CifParser.from_str(mp[r.material_id]).parse_structures(primitive=True)[0]
        e_gen, rel = energy(s, keep=True)
        tsg = sgn(ref, 0.1)
        row = dict(source=r.source, material_id=r.material_id, k=r.k, formula=r.formula, pair_role=r.pair_role,
                   target_sg=tsg, sg_before_01=sgn(s, 0.1), sg_before_001=sgn(s, 0.01), e_gen=e_gen,
                   e_target=ref_energy(r.material_id))
        if rel is not None:
            row.update(sg_after_01=sgn(rel, 0.1), sg_after_001=sgn(rel, 0.01), match_after=bool(SM.fit(rel, ref)))
        hrows.append(row)
        if i % 50 == 0:
            print(f'совпадения: {i}/{len(hits)}', flush=True)
    h = pd.DataFrame(hrows)
    h['d_target'] = h.e_gen - h.e_target
    h.to_csv(a.out.replace('.csv', '_hits.csv'), index=False)
    print(f'\nстрогие совпадения после релаксации ({a.relax_steps} шагов): {len(h)} образцов')
    for role, g in h.groupby('pair_role'):
        print(f'{role:7s} образцов {len(g):4d}  группа цели сохранилась (0,1 Å) {(g.sg_after_01 == g.target_sg).mean():.3f}  '
              f'(0,01 Å) {(g.sg_after_001 == g.target_sg).mean():.3f}  совпадение с целью сохранилось {g.match_after.mean():.3f}  '
              f'медиана E_образца − E_цели {g.d_target.median() * 1000:+.1f} мэВ/атом')

