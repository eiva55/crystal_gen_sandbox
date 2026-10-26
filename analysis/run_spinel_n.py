"""DiffCSP++ на шаблонах шпинели MgAl2O4 (набор configs/csp_targets/spinel_pilot.json): N сэмплов на шаблон, частями (возобновляется).
Шаблоны: spinel_x05_disordered (R3m, неоднозначный) и spinel_x0_ordered (Fd-3m, контроль).
Использование (из корня репозитория):
  nohup python analysis/run_spinel_n.py --label spinel_x05_disordered --n 100 > /dev/null 2>&1 &
Результат: outputs/spinel_n/<label>/<номер>.cif (все части сведены вместе); разбор раскладки — analysis/spinel_decoration.py outputs/spinel_n
"""
import argparse, glob, json, os, shutil, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True, choices=['spinel_x05_disordered', 'spinel_x0_ordered'])
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--chunk', type=int, default=25)
ap.add_argument('--batch-size', type=int, default=20)
ap.add_argument('--ckpt', default='models/diffcsp_pp/checkpoints/pretrained_models/mp_csp')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()

spec = {t['label']: t for t in json.load(open('configs/csp_targets/spinel_pilot.json'))}[a.label]
entry = {k: spec[k] for k in ('atom_types', 'spacegroup_number', 'wyckoff_letters')}
root = os.path.abspath(f'outputs/spinel_n/{a.label}_parts')
final = os.path.abspath(f'outputs/spinel_n/{a.label}')
os.makedirs(root, exist_ok=True)
parts = [min(a.chunk, a.n - i) for i in range(0, a.n, a.chunk)]
for j, size in enumerate(parts):
    out = f'{root}/c{j:02d}'
    if os.path.exists(f'{out}/done.flag'):
        continue
    os.makedirs(out, exist_ok=True)
    json.dump([entry] * size, open(f'{out}/targets.json', 'w'))
    print(f'часть {j + 1}/{len(parts)}: {size} структур', flush=True)
    if a.dry_run:
        continue
    r = subprocess.call(['conda', 'run', '--no-capture-output', '-n', 'diffcsp_pp', 'python', '-u', 'scripts/sample.py',
                         '--model_path', os.path.abspath(a.ckpt), '--save_path', out,
                         '--json_file', f'{out}/targets.json', '--batch_size', str(a.batch_size)], cwd='models/diffcsp_pp')
    if r != 0 or not glob.glob(f'{out}/*.cif'):
        sys.exit(f'часть {j} не завершена (код {r}); запустите ту же команду ещё раз')
    open(f'{out}/done.flag', 'w').write('ok')
if not a.dry_run:
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(final)
    i = 0
    for j in range(len(parts)):
        for p in sorted(glob.glob(f'{root}/c{j:02d}/*.cif'), key=lambda p: int(os.path.basename(p)[:-4])):
            i += 1
            shutil.copy(p, f'{final}/{i}.cif')
    print(f'готово: {i} CIF в {final}')
