import json, collections
import numpy as np
R = {r['name']: r for r in json.load(open('analysis/results/relax_chgnet.json'))}
print('релаксаций:', len(R))
for n in ['ref_mp-352', 'ref_mp-685097', 'ref_mp-3536', 'ref_mp-34330']:
    r = R[n]
    print(f"{n}: E {r['E0']:.4f} -> {r['E1']:.4f}, SG до {r['sg_before']} после {r['sg_after']}, шагов {r['steps']}")
Es, Em = R['ref_mp-352']['E1'], R['ref_mp-685097']['E1']
print(f"зазор meta-stable после релаксации CHGNet: {Em - Es:.4f} эВ/атом (MP: ~0.02-0.03)")
for lab in ['hfo2_stable', 'hfo2_meta']:
    rs = [r for k, r in R.items() if k.startswith(lab + '/')]
    print(f"\n{lab}: N={len(rs)}")
    for key in ('sg_before', 'sg_after'):
        for tol in ('0.01', '0.1'):
            print(f"  {key} symprec={tol}:", dict(collections.Counter(r[key][tol] for r in rs)))
    d = np.array([r['E1'] - Es for r in rs])
    print(f"  E(релакс.) - E(релакс. mp-352): среднее {d.mean():.4f}, мин {d.min():.4f}, макс {d.max():.4f} эВ/атом")
for lab in ['spinel_x0_ordered', 'spinel_x05_disordered']:
    rs = [r for k, r in R.items() if k.startswith(lab + '/')]
    print(f"\n{lab}: N={len(rs)}, E1 среднее {np.mean([r['E1'] for r in rs]):.4f}, "
          f"мин {min(r['E1'] for r in rs):.4f}, макс {max(r['E1'] for r in rs):.4f}")
