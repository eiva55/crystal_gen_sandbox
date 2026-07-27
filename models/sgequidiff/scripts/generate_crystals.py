from datetime import datetime
import math
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
import time

from omegaconf import OmegaConf

from aliases import *
from crystal_classes import ASUCrystal
from model.crystal_sampler import CrystalSampler
from utils.train_utils import dispatch_model, experiment_setup
from utils.data_utils import asu_to_pymatgen_structure, pack_and_save
from utils.io_utils import convert_wandb_config_to_hydra_config

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--outfile_name", type=str, default="")
parser.add_argument("--num_samples", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--save_method", type=str, default="npz")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--load_best_submodules", action="store_true")
parser.add_argument("--space_group_number", type=int, default=None)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args(sys.argv[1:])


def get_checkpoint_path(ckpt_dir: str) -> str:
    """Load most recent snapshot from 'ckpt_dir'."""
    snapshot_paths: List[Path] = list(Path(ckpt_dir).glob("**/*snapshot-*"))
    assert len(snapshot_paths) > 0

    snapshot_versions: List[int] = [
        int(snapshot.with_suffix("").as_posix().split("-")[-1])
        for snapshot in snapshot_paths
    ]
    most_recent_snapshot: Path = snapshot_paths[
        max(enumerate(snapshot_versions), key=lambda x: x[1])[0]
    ]
    return most_recent_snapshot.as_posix()


def load_model_from_ckpt_(
    model: CrystalSampler,
    ckpt_dir: str,
    device: torch.device,
    load_best_submodules: bool = False,
):
    if load_best_submodules:
        submodule_snapshot_paths: List[Path] = list(Path(ckpt_dir).glob("**/best*snapshot*"))
        assert len(submodule_snapshot_paths) > 0
        for snapshot_path in submodule_snapshot_paths:
            path_str: str = snapshot_path.as_posix()
            state_dict = torch.load(path_str, map_location=device)
            if "best_space_group_snapshot.pt" in path_str:
                model.space_group_sampler.load_state_dict(state_dict)
            elif "best_lattice_snapshot.pt" in path_str:
                model.lattice_sampler.load_state_dict(state_dict)
            elif "best_wyckoff-transformer_snapshot.pt" in path_str:
                model.wyckoff_and_element_sampler.load_state_dict(state_dict)
            elif "best_diffusion_snapshot.pt" in path_str:
                model.atom_coord_diffusion_model.load_state_dict(state_dict)
    else:
        checkpoint_path: str = get_checkpoint_path(ckpt_dir)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])


@torch.no_grad()
def main(
    ckpt_dir: str,
    num_samples: int = 10,
    outfile_name: str = '',
    save_method: str = 'poscar',
    batch_size: int = 32,
    load_best_submodules: bool = False,
):
    device_is_cuda: bool = torch.cuda.is_available()
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    config_path: str = (Path(ckpt_dir) / "config.yaml").as_posix()
    config: DictConfig = convert_wandb_config_to_hydra_config(config_path)
    if args.space_group_number is not None:
        space_group_numbers = torch.tensor(
            [args.space_group_number], dtype=torch.long, device=device
        ).expand(batch_size)
    else:
        space_group_numbers = None

    # asu dictionary, embeddings, and seeding
    experiment_setup(config)

    # Load model
    model: CrystalSampler = dispatch_model(config, device)
    load_model_from_ckpt_(model, ckpt_dir, device, load_best_submodules)
    model.to(device)
    model.eval()

    # Generate crystals
    batch_size = min(num_samples, batch_size)
    asu_crystals: List[ASUCrystal] = []
    start_time = time.perf_counter()
    for i in tqdm(range(math.ceil(num_samples / batch_size))):
        if i == 1:
            if device_is_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

        samples: List[ASUCrystal] = model.sample_crystal(
            batch_size,
            diffusion_snr=0.4,
            temperature=args.temperature,
            space_group_numbers=space_group_numbers,
            lattice_parameters=None,
        )
        asu_crystals[0:0] = samples  # fast way to prepend a list
    if device_is_cuda:
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    asu_crystals = asu_crystals[:num_samples]
    print(f"Generated {len(asu_crystals)} crystals. Elapsed time (s): {end_time-start_time}")

    if args.verbose:
        for i in range(num_samples):
            print(f"------- {i} -------")
            print(f"Lattice: {asu_crystals[i].conventional_lattice_lengths, asu_crystals[i].conventional_lattice_angles}")
            print(asu_crystals[i])

    # Save crystals
    if save_method == "poscar":
        structures: List[pmg_structure] = [asu_to_pymatgen_structure(asu) for asu in asu_crystals]
        for i, structure in enumerate(structures):
            structure.to((Path(ckpt_dir) / f"POSCAR-{i}-{outfile_name}").as_posix(), fmt="poscar")
    elif save_method == "npz":
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename: Path = Path(ckpt_dir) / f"{current_time}_{outfile_name}_{num_samples}samples_{str(args.temperature).replace('.', '-')}T"
        pack_and_save(crystals=asu_crystals, save_path=filename.as_posix())
    else:
        raise AttributeError

    print(f"Generated crystals saved in {Path(ckpt_dir).as_posix()}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main(
        ckpt_dir=args.ckpt_dir,
        num_samples=args.num_samples,
        outfile_name=args.outfile_name,
        save_method=args.save_method,
        batch_size=args.batch_size,
        load_best_submodules=args.load_best_submodules,
    )
