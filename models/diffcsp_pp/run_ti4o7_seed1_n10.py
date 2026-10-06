import sys
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from eval_utils import load_model, lattices_to_params_shape
from diffcsp.pl_data.dataset import CrystDataset

MODEL_PATH = Path("/home/bebra/matgen/crystal_gen_sandbox/models/diffcsp_pp/checkpoints/pretrained_models/mp_csp")
CSV_PATH = "/home/bebra/matgen/crystal_gen_sandbox/models/diffcsp_pp/data/ti4o7_seed1_n10.csv"
SAVE_PATH = "/tmp/ti4o7_seed1_n10_sym.pt"

model, _, cfg = load_model(MODEL_PATH, load_data=False)
if torch.cuda.is_available():
    model.to("cuda")

dataset = CrystDataset(
    name="ti4o7_seed1_n10", path=CSV_PATH, save_path=SAVE_PATH,
    prop=cfg.data.prop, niggli=cfg.data.niggli, primitive=cfg.data.primitive,
    graph_method=cfg.data.graph_method, preprocess_workers=1,
    lattice_scale_method=cfg.data.lattice_scale_method,
    tolerance=cfg.data.tolerance, use_space_group=cfg.data.use_space_group,
    use_pos_index=cfg.data.use_pos_index,
)
loader = PyGDataLoader(dataset, batch_size=10, shuffle=False)
batch = next(iter(loader))
if torch.cuda.is_available():
    batch = batch.cuda()

print("Starting diffusion sampling (seed1, N=10)...")
outputs, traj = model.sample(batch, step_lr=1e-5)

torch.save({
    'frac_coords': outputs['frac_coords'].detach().cpu(),
    'atom_types': outputs['atom_types'].detach().cpu(),
    'lattices': outputs['lattices'].detach().cpu(),
    'num_atoms': outputs['num_atoms'].detach().cpu(),
    'material_ids': dataset.df['material_id'].tolist(),
}, '/tmp/ti4o7_seed1_n10_output.pt')
print("Saved to /tmp/ti4o7_seed1_n10_output.pt")
