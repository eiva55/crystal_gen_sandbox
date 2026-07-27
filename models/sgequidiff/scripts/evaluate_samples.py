import pickle
from datetime import datetime
import argparse
import sys
import multiprocessing
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import warnings

import torch
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from p_tqdm import p_map
import matplotlib.pyplot as plt
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ase.io

from utils.io_utils import get_project_dir, DATA_DIRECTORY
from utils.eval_utils import (
    CompositionScaler,
    Crystal,
    get_fingerprint_pairwise_dist,
    compute_cov,
    get_fingerprint_central_moment_discrepancy,
    get_unique_structures,
    get_novel_structures,
    load_dataset_pymatgen_structures,
    HiddenPrints,
)
from utils.data_utils import (
    batched_convert_asu_frac_coords_to_primitive_cartesian_coords,
    unpack,
    lattice_params_to_matrix_torch,
    lattice_matrix_to_params,
)
from aliases import *
from crystal_classes import ASUCrystal
import global_vars

parser = argparse.ArgumentParser()
parser.add_argument("--gen_crystals_path", type=str, required=True)
parser.add_argument(
    "--model_name", type=str, default="sgequidiff",
    choices=["cdvae", "sgequidiff", "diffcsp-pp", "diffcsp", "symmcd", "mattergen"]
)
parser.add_argument(
    "--dataset_name", required=True, default=None, type=str, choices=["mp20", "mpts52"]
)
parser.add_argument("--max_limit_cpus", type=int, default=16)
args = parser.parse_args(sys.argv[1:])
dump_dir: str = Path(args.gen_crystals_path).parent.as_posix()
num_cpus = min(args.max_limit_cpus, multiprocessing.cpu_count())
print(f"{multiprocessing.cpu_count()} CPUs found, using {num_cpus}.")

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class MatterGenDataReader:
    def __init__(self):
        pass

    def load_structures(self, zipped_cifs_path: str) -> Sequence[Structure]:
        input_path = Path(zipped_cifs_path)
        if input_path.suffix == ".zip":
            with TemporaryDirectory() as tmpdirname:
                with ZipFile(input_path, "r") as zip_obj:
                    zip_obj.extractall(tmpdirname)
                return self.extract_structures_from_folder(tmpdirname)

    def extract_structures_from_folder(self, dirname: str) -> Sequence[Structure]:
        structures = []
        for filename in os.listdir(dirname):
            if filename.endswith(".cif"):
                try:
                    with HiddenPrints():
                        structures.append(Structure.from_file(f"{dirname}/{filename}"))
                except ValueError as e:
                    warnings.warning(f"Failed to read {filename} as a CIF file: {e}")
            elif filename.endswith(".extxyz") or filename.endswith(".xyz"):
                ase_atoms = ase.io.read(
                    f"{dirname}/{filename}", 0
                )  #  We assume that the file contains only one structure
                with HiddenPrints():
                    structures.append(AseAtomsAdaptor.get_structure(ase_atoms))
        return structures


class DataReader:
    def __init__(self, model_name: str):
        assert model_name in ["cdvae", "diffcsp-pp", "diffcsp", "symmcd"]
        self.model_name = model_name

    def load_data(self, file_path):
        if file_path[-3:] == 'npy':
            data = np.load(file_path, allow_pickle=True).item()
            for k, v in data.items():
                if k == 'input_data_batch':
                    for k1, v1 in data[k].items():
                        data[k][k1] = torch.from_numpy(v1)
                else:
                    data[k] = torch.from_numpy(v).unsqueeze(0)
        else:
            data = torch.load(file_path, map_location='cpu')
        return data

    def get_crystals_list(
        self,
        frac_coords: ndarray,
        atom_types: ndarray,
        lengths: ndarray,
        angles: ndarray,
        num_atoms: ndarray
    ):
        """
        args:
            frac_coords: (num_atoms, 3)
            atom_types: (num_atoms) array of atomic numbers (1-indexed)
            lengths: (num_crystals)
            angles: (num_crystals)
            num_atoms: (num_crystals)
        """
        assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
        assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

        start_idx = 0
        crystal_array_list = []
        for batch_idx, num_atom in enumerate(num_atoms.tolist()):
            cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
            cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
            cur_lengths = lengths[batch_idx]
            cur_angles = angles[batch_idx]

            crystal_array_list.append({
                'frac_coords': cur_frac_coords.detach().cpu().numpy(),
                'atom_types': cur_atom_types.detach().cpu().numpy(),
                'lengths': cur_lengths.detach().cpu().numpy(),
                'angles': cur_angles.detach().cpu().numpy(),
            })
            start_idx = start_idx + num_atom
        return crystal_array_list

    def get_crystal_array_list(
            self, file_path, batch_idx=0
    ) -> Tuple[List[dict], List[dict]]:
        data = self.load_data(file_path)
        if self.model_name == "cdvae":
            crys_array_list = self.get_crystals_list(
                data['frac_coords'][batch_idx],
                data['atom_types'][batch_idx],
                data['lengths'][batch_idx],
                data['angles'][batch_idx],
                data['num_atoms'][batch_idx]
            )
        elif self.model_name in ["diffcsp-pp", "diffcsp", "symmcd"]:
            crys_array_list = self.get_crystals_list(
                data['frac_coords'],
                data['atom_types'],
                data['lengths'],
                data['angles'],
                data['num_atoms'],
            )

        if 'input_data_batch' in data:
            # Load ground truth crystals
            batch = data['input_data_batch']
            if isinstance(batch, dict):
                true_crystal_array_list = self.get_crystals_list(
                    batch['frac_coords'], batch['atom_types'], batch['lengths'],
                    batch['angles'], batch['num_atoms'])
            else:
                true_crystal_array_list = self.get_crystals_list(
                    batch.frac_coords, batch.atom_types, batch.lengths,
                    batch.angles, batch.num_atoms)
        else:
            true_crystal_array_list = None
        return crys_array_list, true_crystal_array_list


class GenEval(object):
    def __init__(
            self,
            pred_crys: Iterable[Crystal],
            test_crys: Iterable[Crystal],
            train_crys: Iterable[pmg_structure],  # for evaluating novelty
            dataset_name: str = None,
            model_name: str = None,
            n_samples: int = 1000,
            device: Union[str, torch.device] = 'cpu',
    ):
        """
        Args:
            pred_crys: Generated crystals.
            test_crys: Ground truth crystals.
            n_samples: Required number of generated crystals which are deemed
                structurally and chemically valid.
            dataset_name:
            model_name: Filename for pickle of space groups and Wyckoff
                dimensions of generated and ground truth crystals. For plotting
                purposes.
        """
        assert dataset_name in ['mp20', 'mpts52']
        self.crys: Iterable[Crystal] = pred_crys
        self.test_crys: Iterable[Crystal] = test_crys
        self.train_crys: Iterable[pmg_structure] = train_crys
        self.n_samples: int = n_samples
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        # self.current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Get samples which are structurally and compositionally valid
        valid_crys = [c for c in pred_crys if c.valid]
        print(f'Number of structurally valid crystals: {len([c for c in pred_crys if c.struct_valid])}/{len(pred_crys)}')
        print(f'Number of valid crystals: {len(valid_crys)}/{len(pred_crys)}')
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            n_constructed = len([1 for crystal in pred_crys if crystal.constructed])
            print(f'n_constructed: {n_constructed}/{len(pred_crys)}')
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

        # Of the valid samples, filter out non-unique ones
        structure_matcher = StructureMatcher()  # MatterGen Novelty settings
        unique_sample_idxs: List[int] = get_unique_structures(
            [c.structure for c in self.valid_samples], structure_matcher
        )[1]
        unique_valid_samples: List[Crystal] = [self.valid_samples[i] for i in unique_sample_idxs]
        print(f"{len(unique_valid_samples)} unique samples out of {len(self.valid_samples)} valid samples.")

        # Of the remaining samples, filter out those found in the training data
        novel_unique_valid_sample_idxs = get_novel_structures(
            [c.structure for c in unique_valid_samples], train_crys, structure_matcher
        )[1]
        self.valid_un_samples: List[Crystal] = [
            unique_valid_samples[i] for i in novel_unique_valid_sample_idxs
        ]
        print(f"{len(self.valid_un_samples)} U.N. samples out of {len(self.valid_samples)} valid samples.")
        self.unique_rate = float(len(unique_valid_samples)) / len(self.valid_samples)
        self.un_rate = float(len(self.valid_un_samples)) / len(self.valid_samples)
        with open(f"{dump_dir}/valid_un_samples_{model_name}_{dataset_name}.pkl", "wb") as f:
            pickle.dump(self.valid_un_samples, f)

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,       # valid composition
                'struct_valid': struct_valid,   # valid structure
                'valid': valid}

    def get_composition_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_un_samples]
        comp_fps = CompositionScaler.transform(comp_fps)
        comp_div = get_fingerprint_pairwise_dist(comp_fps)
        return {'comp_diversity': comp_div}

    def get_structural_diversity(self):
        return {'struct_diversity': get_fingerprint_pairwise_dist([c.struct_fp for c in self.valid_un_samples if c.struct_fp is not None])}

    def get_structural_fingerprint_divergence(self):
        generated_structure_fingerprints = torch.tensor(
            np.array(
                [c.struct_fp for c in self.crys if c.struct_fp is not None]
            ), device=self.device
        )
        ground_truth_structure_fingerprints = torch.tensor(
            np.array([c.struct_fp for c in self.test_crys if c.struct_fp is not None]), device=self.device
        )
        return {
            'structural_divergence': get_fingerprint_central_moment_discrepancy(
                generated_structure_fingerprints,
                ground_truth_structure_fingerprints,
                n_moments=50,
            )
        }

    def get_chemical_fingerprint_divergence(self):
        generated_chemical_fingerprints = torch.tensor(
            [c.comp_fp for c in self.crys if c.comp_fp is not None], device="cpu"
        )
        ground_truth_chemical_fingerprints = torch.tensor(
            [c.comp_fp for c in self.test_crys if c.comp_fp is not None], device="cpu"
        )
        generated_chemical_fingerprints = CompositionScaler.transform(
            generated_chemical_fingerprints
        )
        ground_truth_chemical_fingerprints = CompositionScaler.transform(
            ground_truth_chemical_fingerprints
        )

        return {
            'chemical_divergence': get_fingerprint_central_moment_discrepancy(
                generated_chemical_fingerprints,
                ground_truth_chemical_fingerprints,
                n_moments=10,
            )
        }

    def get_density_wasserstein_distance(self):
        """Get distance to target marginal of crystal densities in g/cm^3"""
        pred_densities = [c.structure.density for c in self.valid_un_samples]
        gt_densities = [c.structure.density for c in self.test_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        """Get distance to target marginal of number of elements per crystal"""
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_un_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.test_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.dataset_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.test_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_space_group_marginals_jsd(self):
        pred_space_groups = [c.space_group_number for c in self.valid_un_samples]
        gt_space_groups = [c.space_group_number for c in self.test_crys]

        _bin_edges = np.array([i - 0.5 for i in range(1, 232)])
        pred_space_group_probs = np.histogram(np.array(pred_space_groups), bins=_bin_edges, density=True)[0]
        gt_space_group_probs = np.histogram(np.array(gt_space_groups), bins=_bin_edges, density=True)[0]
        jsd_space_groups = jensenshannon(pred_space_group_probs, gt_space_group_probs)
        if self.model_name is not None:
            # Save data for plotting later
            with open(
                f"{dump_dir}/{self.dataset_name}_{self.model_name}_space_groups.pkl",
                "wb"
            ) as f:
                pickle.dump(pred_space_groups, f)
            if not os.path.isfile(
                f"{dump_dir}/{self.dataset_name}_gt_space_groups.pkl"
            ):
                with open(
                    f"{dump_dir}/{self.dataset_name}_gt_space_groups.pkl",
                    "wb"
                ) as f:
                    pickle.dump(gt_space_groups, f)
        return {'jsd_space_groups': jsd_space_groups}

    def get_wyckoff_dimensionality_marginals_jsd(self):
        pred_wyckoff_dims = np.concatenate(
            [c.wyckoff_dims for c in self.valid_un_samples], axis=0
        )
        gt_wyckoff_dims = np.concatenate(
            [c.wyckoff_dims for c in self.test_crys], axis=0
        )

        _bin_edges = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
        pred_wyckoff_dim_probs = np.histogram(np.array(pred_wyckoff_dims), bins=_bin_edges, density=True)[0]
        gt_wyckoff_dim_probs = np.histogram(np.array(gt_wyckoff_dims), bins=_bin_edges, density=True)[0]
        jsd_wyckoff_dims = jensenshannon(pred_wyckoff_dim_probs, gt_wyckoff_dim_probs)
        if self.model_name is not None:
            # Save data for plotting later
            with open(
                f"{dump_dir}/{self.dataset_name}_{self.model_name}_wyck_dims.pkl",
                "wb"
            ) as f:
                pickle.dump(pred_wyckoff_dims, f)
            if not os.path.isfile(
                f"{dump_dir}/{self.dataset_name}_gt_wyck_dims.pkl"
            ):
                with open(
                    f"{dump_dir}/{self.dataset_name}_gt_wyck_dims.pkl",
                    "wb"
                ) as f:
                    pickle.dump(gt_wyckoff_dims, f)
        return {'jsd_wyckoff_dimensions': jsd_wyckoff_dims}

    def get_template_statistics(self):
        """
        Following SymmCD, compute the following for generated samples:
            - number of unique templates
            - number of templates not in the training set
        Unlike SymmCD, we will only consider templates of structurally and
        compositionally valid crystals.
        """
        assert self.dataset_name is not None
        pickled_train_crystals_path: str = (
            DATA_DIRECTORY / (self.dataset_name + "_train_crystals.pkl")
        ).as_posix()
        if os.path.exists(pickled_train_crystals_path):
            with open(pickled_train_crystals_path, "rb") as f:
                train_crystals: List[Crystal] = pickle.load(f)
        else:
            train_crystal_dicts: List[Dict] = [
                {
                    "frac_coords": s.frac_coords,
                    "atom_types": np.array([e.Z for e in s.species]),
                    "lengths": np.array(s.lattice.lengths),
                    "angles": np.array(s.lattice.angles),
                } for s in self.train_crys
            ]
            train_crystals: List[Crystal] = p_map(
                lambda x: Crystal(x), train_crystal_dicts, num_cpus=num_cpus
            )
            with open(pickled_train_crystals_path, "wb") as f:
                pickle.dump(train_crystals, f, protocol=pickle.HIGHEST_PROTOCOL)

        pred_templates = [c.immutable_template for c in self.valid_samples]
        train_templates = [c.immutable_template for c in train_crystals]

        unique_pred_templates = set(pred_templates)
        unique_train_templates = set(train_templates)

        num_unique_templates = len(unique_pred_templates)
        num_novel_templates = sum([
            1 for pred_template in unique_pred_templates
            if pred_template not in unique_train_templates
        ])
        return {
            f"num unique templates / {len(self.valid_samples)}": num_unique_templates,
            f"num novel templates / {len(self.valid_samples)}": num_novel_templates,
        }

    @torch.no_grad()
    def get_metrics(self) -> Dict:
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_composition_diversity())
        metrics.update(self.get_structural_diversity())
        metrics.update(self.get_density_wasserstein_distance())
        metrics.update(self.get_num_elem_wdist())
        # metrics.update(self.get_prop_wdist())
        metrics.update(self.get_space_group_marginals_jsd())
        metrics.update(self.get_wyckoff_dimensionality_marginals_jsd())
        metrics.update(self.get_structural_fingerprint_divergence())
        metrics.update(self.get_chemical_fingerprint_divergence())
        metrics.update({"unique_rate": self.unique_rate, "U.N. rate": self.un_rate})
        if 'mpts' not in self.dataset_name:
            metrics.update(self.get_coverage())
        metrics.update(self.get_template_statistics())
        print(metrics)
        return metrics


def convert_asu_crystals_to_primitive_crystals(
    asu_crystals: List[ASUCrystal]
) -> Iterable[Crystal]:
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Pack crystals
    space_group_numbers: List[Tensor] = []  # 1-indexed
    conventional_lattice_lengths: List[Tensor] = []
    conventional_lattice_angles: List[Tensor] = []
    wyckoff_indices: List[Tensor] = []
    asu_atom_type_indices: List[Tensor] = []  # 0-indexed
    asu_frac_coords: List[Tensor] = []
    n_coords_per_asu: List[int] = []
    prim_frac_coords = []
    prim_element_indices = []
    prim_wyckoff_indices = []
    num_prim_atoms_per_crystal = []
    prim_lattice_lengths = []
    prim_lattice_angles = []
    for crystal in asu_crystals:
        assert crystal.num_atoms > 0
        crystal = crystal.to(device)
        space_group_numbers.append(crystal.space_group_number)
        conventional_lattice_lengths.append(crystal.conventional_lattice_lengths)
        conventional_lattice_angles.append(crystal.conventional_lattice_angles)
        frac_coord = crystal.conventional_frac_coords
        wyckoff_indices.append(crystal.wyckoff_indices)
        asu_atom_type_indices.append(crystal.element_indices)
        asu_frac_coords.append(frac_coord)
        n_coords_per_asu.append(frac_coord.shape[0])

        (
            xtal_prim_frac_coords,               # (n_primitive_atoms, 3)
            xtal_prim_element_indices,           # (n_primitive_atoms,)
            xtal_prim_wyckoff_indices,           # (n_primitive_atoms,)
            xtal_num_prim_atoms_per_crystal,     # (n_crystals,)
            xtal_prim_lattice_matrix,            # (n_crystals, 3, 3)
            _, _,
        ) = batched_convert_asu_frac_coords_to_primitive_cartesian_coords(
            asu_frac_coords=crystal.conventional_frac_coords,
            asu_wyckoff_indices=crystal.wyckoff_indices,
            asu_element_indices=crystal.element_indices,
            n_coords_per_asu=torch.tensor([crystal.num_atoms], dtype=torch.long, device=device),
            conventional_lattice_matrix=lattice_params_to_matrix_torch(
                crystal.conventional_lattice_lengths.view(1, 3),
                crystal.conventional_lattice_angles.view(1, 3),
            ),
            space_group_indices=torch.tensor([crystal.space_group_number-1], dtype=torch.long, device=device),
            device=device,
            return_cartesian_coords=False
        )
        prim_lengths, prim_angles = lattice_matrix_to_params(xtal_prim_lattice_matrix)
        prim_frac_coords.append(xtal_prim_frac_coords)
        prim_element_indices.append(xtal_prim_element_indices)
        prim_wyckoff_indices.append(xtal_prim_wyckoff_indices)
        num_prim_atoms_per_crystal.append(xtal_num_prim_atoms_per_crystal)
        prim_lattice_lengths.append(prim_lengths)
        prim_lattice_angles.append(prim_angles)

    prim_frac_coords = torch.cat(prim_frac_coords)
    prim_element_indices = torch.cat(prim_element_indices)
    num_prim_atoms_per_crystal = torch.cat(num_prim_atoms_per_crystal)
    prim_lattice_lengths = torch.cat(prim_lattice_lengths).cpu()
    prim_lattice_angles = torch.cat(prim_lattice_angles).cpu()

    space_group_numbers: Tensor = torch.tensor(space_group_numbers, dtype=torch.long, device=device)
    space_group_indices = space_group_numbers - 1
    # (n_crystals,)
    wyckoff_indices: Tensor = torch.cat(wyckoff_indices, dim=0)
    n_coords_per_asu_tensor = torch.tensor(n_coords_per_asu, dtype=torch.long, device=device)

    asu_wyckoff_dimensionalities: tensor = global_vars.wyckoff_dimension_tensor.to(device)[
        space_group_indices.repeat_interleave(n_coords_per_asu_tensor), wyckoff_indices
    ]

    # Unpack tensors with torch.split()
    asu_wyckoff_dimensions: Tuple[Tensor] = torch.split(
        asu_wyckoff_dimensionalities.cpu(), n_coords_per_asu, dim=0
    )  # len-n_crystals
    num_prim_atoms_per_crystal = num_prim_atoms_per_crystal.tolist()
    prim_frac_coords: Tuple[Tensor] = torch.split(
        prim_frac_coords.cpu(), num_prim_atoms_per_crystal, dim=0
    )  # len-n_crystals tuple
    prim_atomic_numbers: Tuple[Tensor] = torch.split(
        (prim_element_indices + 1).cpu(), num_prim_atoms_per_crystal, dim=0
    )  # len-n_crystals tuple

    def tensors_to_Crystal(
        space_group_number: ndarray,
        crystal_frac_coords: Tensor,
        crystal_atomic_numbers: Tensor,
        asu_crystal_wyckoff_dimensions: Tensor,
        lattice_lengths: Tensor,
        lattice_angles: Tensor,
    ):
        return Crystal({
            'frac_coords': crystal_frac_coords.numpy(),
            'space_group_number': int(space_group_number),
            'atom_types': crystal_atomic_numbers.numpy(),
            'wyckoff_dims': asu_crystal_wyckoff_dimensions.numpy(),
            'lengths': lattice_lengths,
            'angles': lattice_angles,
        })

    crystals: Iterable[Crystal] = p_map(
        tensors_to_Crystal,
        space_group_numbers.cpu().numpy(),
        prim_frac_coords,
        prim_atomic_numbers,
        asu_wyckoff_dimensions,
        prim_lattice_lengths,
        prim_lattice_angles,
        num_cpus=num_cpus,
    )
    return crystals


@torch.no_grad()
def evaluate_generated_crystals(
    generated_crystals_filepath: str,
    model_name: str = 'cdvae',
    dataset_name: str = 'mp20',
    device: Union[str, torch.device] = 'cpu',
):
    # -- Load crystals and convert them to pymatgen structures
    train_crystals: List[pmg_structure] = load_dataset_pymatgen_structures(
        dataset_name=dataset_name, split="train",
    )

    # Load test crystals
    print("Loading test crystals.")
    pickled_test_crystals_path = (DATA_DIRECTORY / (dataset_name + "_test_crystals.pkl")).as_posix()
    if os.path.exists(pickled_test_crystals_path):
        with open(pickled_test_crystals_path, "rb") as f:
            test_crystals: List[Crystal] = pickle.load(f)
    else:
        test_structures: List[pmg_structure] = load_dataset_pymatgen_structures(
            dataset_name=dataset_name, split="test",
        )
        test_crystal_dicts: List[dict] = [
            {
                "frac_coords": s.frac_coords,
                "atom_types": np.array([e.Z for e in s.species]),
                "lengths": np.array(s.lattice.lengths),
                "angles": np.array(s.lattice.angles),
            } for s in test_structures
        ]
        test_crystals: List[Crystal] = p_map(
            lambda x: Crystal(x), test_crystal_dicts, num_cpus=num_cpus
        )
        with open(pickled_test_crystals_path, "wb") as f:
            pickle.dump(test_crystals, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Successfully loaded test crystals.")

    print("Loading generated crystals.")
    if model_name in ['cdvae', 'diffcsp-pp', 'diffcsp', 'symmcd']:
        cdvae_data_reader = DataReader(model_name)
        generated_crystal_dicts: List[dict] = (
            cdvae_data_reader.get_crystal_array_list(generated_crystals_filepath)[0]
        )
        generated_crystals: Iterable[Crystal] = p_map(
            lambda x: Crystal(
                x,
                convert_to_primitive=(model_name == 'diffcsp-pp'),
                atom_types_are_logits=(model_name == 'diffcsp'),
            ),
            generated_crystal_dicts,
            num_cpus=num_cpus,
        )
    elif model_name == "mattergen":
        data_reader = MatterGenDataReader()
        gen_structures: Sequence[Structure] = data_reader.load_structures(
            generated_crystals_filepath
        )
        generated_crystals: Iterable[Crystal] = p_map(
            lambda x: Crystal({}, False, False, x),
            gen_structures,
            num_cpus=num_cpus,
        )
    elif model_name == 'sgequidiff':
        # Load ASUCrystals from file
        generated_asu_crystals: List[ASUCrystal] = unpack(generated_crystals_filepath)
        # TODO: standardize this
        if dataset_name == "mp20":
            _dataset_name = "mp_20"
        elif dataset_name == "mpts52":
            _dataset_name = "mpts_52"
        else:
            _dataset_name = dataset_name

        generated_crystals: Iterable[Crystal] = (
            convert_asu_crystals_to_primitive_crystals(generated_asu_crystals)
        )
    else:
        raise NotImplementedError

    print("All crystals loaded. Beginning evaluation.")
    evaluator = GenEval(
        generated_crystals,
        test_crystals,   # to compute test statistics
        train_crystals,  # to measure novelty
        n_samples=1000,
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
    )
    evaluation_metrics: Dict = evaluator.get_metrics()

    # Save metrics to json
    filename: str = Path(args.gen_crystals_path).with_suffix("").as_posix() + "_EVAL_METRICS.json"
    with open(filename, "w") as f:
        json.dump(evaluation_metrics, f)


if __name__ == '__main__':
    PROJECT_DIR = get_project_dir()
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    generated_crystals_filepath = args.gen_crystals_path
    model_name = args.model_name
    dataset_name = args.dataset_name
    if dataset_name not in ["mp20", "mpts52"]:
        raise NotImplementedError

    evaluate_generated_crystals(
        generated_crystals_filepath=generated_crystals_filepath,
        model_name=model_name,
        dataset_name=dataset_name,
        device=device,
    )
