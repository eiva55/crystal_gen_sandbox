import torch
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import sys
sys.path.insert(0, "scripts")
from eval_utils import get_crystals_list

data = torch.load('/tmp/magneli_eval_v3_output.pt')
frac_coords = data['frac_coords']
atom_types = data['atom_types']
lattices = data['lattices']
num_atoms = data['num_atoms']
material_ids = data['material_ids']

from eval_utils import lattices_to_params_shape
lengths, angles = lattices_to_params_shape(lattices)

crystals = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)

for idx, crystal in enumerate(crystals):
    label = material_ids[idx] if idx < len(material_ids) else f"idx{idx}"
    try:
        s = Structure(
            lattice=Lattice.from_parameters(*crystal['lengths'].tolist(), *crystal['angles'].tolist()),
            species=crystal['atom_types'].tolist(),
            coords=crystal['frac_coords'].tolist(),
            coords_are_cartesian=False,
        )
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        sg = sga.get_space_group_symbol()
        sg_num = sga.get_space_group_number()
        print(f"{label}: {s.composition.reduced_formula}, SG={sg} (#{sg_num}), a={s.lattice.a:.2f}, b={s.lattice.b:.2f}, c={s.lattice.c:.2f}")
        s.to(filename=f"/tmp/magneli_v3_{label}.cif")
    except Exception as e:
        print(f"{label}: FAILED — {e}")
