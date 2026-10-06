from diffcsp.common.data_utils import preprocess
import torch

result = preprocess(
    '/home/bebra/matgen/crystal_gen_sandbox/models/diffcsp_pp/data/ti4o7_seeded_templates.csv',
    num_workers=1, niggli=True, primitive=False, graph_method='crystalnn',
    prop_list=['formation_energy_per_atom'], use_space_group=True, tol=0.1,
)
torch.save(result, '/tmp/ti4o7_seeded_sym.pt')
for entry in result:
    print(entry['mp_id'], '-> spacegroup:', entry.get('spacegroup', 'MISSING'))
