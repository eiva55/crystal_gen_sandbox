"""Conversion script for the MP-20 dataset: converts the provided CIF 
representations into ASUCrystal objects.
"""
import argparse
import csv
import logging
from pathlib import Path
import multiprocessing
from typing import List
import functools
import sys
import os

from pymatgen.core.structure import Structure
from pyxtal.util import symmetrize
import pandas as pd

from crystal_classes import ASUCrystal
from utils.data_utils import pyxtal_pymatgen_structure_to_asu, pack_and_save
from utils.io_utils import DATA_DIRECTORY, save_object, load_object, extract, remove_suffix
from utils.logging_utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mp20", choices=["mp20", "mpts52"])
parser.add_argument("--assume_p1_only", action="store_true")  # default False


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def process_csv_entry(args, destination: Path, entry: dict, dataset_name: str) -> None:
    cif: str = entry["cif"]
    with HiddenPrints():
        pymatgen_structure: Structure = Structure.from_str(cif, fmt="cif")
        pymatgen_canonical_structure, _ = symmetrize(
            pymatgen_structure, tol=0.1, a_tol=5.0, style="pyxtal"
        )
        crystal: ASUCrystal = pyxtal_pymatgen_structure_to_asu(
            pymatgen_canonical_structure,
            assume_p1_only=args.assume_p1_only,
            structure_to_fall_back_to=pymatgen_structure,
        )
    if dataset_name == "mpts52":
        properties = {
            "material_id": entry.get("material_id", None),
            "formation_energy_per_atom": entry.get("formation_energy_per_atom", None),
            "energy_above_hull": entry.get("energy_above_hull", None),
        }
    elif dataset_name == "mp20":
        properties = {
            "material_id": entry.get("material_id", None),
            "formation_energy_per_atom": entry.get("formation_energy_per_atom", None),
            "band_gap": entry.get("band_gap", None),
            "e_above_hull": entry.get("e_above_hull", None),
        }
    else:
        raise NotImplementedError
    save_object((crystal, properties), destination)  # pickles ASUCrystal


def process_queue(
    args, queue: multiprocessing.Queue, lock: multiprocessing.Lock, destination: Path,
):
    while True:
        entry: dict = queue.get()
        if entry is None:
            break
        material_id: str = entry["material_id"]
        with lock:
            pass
        process_csv_entry(args, destination / material_id, entry, args.dataset)


def _initializer(args, destination, q, l):
    process_queue(args, q, l, destination)


def process_csv(log: logging.Logger, args, csv_location: Path, destination: Path) -> None:
    num_processors: int = multiprocessing.cpu_count()
    handle = open(csv_location, newline="")
    entries: List[dict] = list(csv.DictReader(handle))
    handle.close()

    work_queue = multiprocessing.Queue(maxsize=num_processors)
    lock: multiprocessing.Lock = multiprocessing.Lock()

    _init_fn = functools.partial(_initializer, args, destination)
    pool = multiprocessing.Pool(
        num_processors,
        initializer=_init_fn,
        initargs=(work_queue, lock),
    )

    for entry in entries:
        work_queue.put(entry)

    for _ in range(num_processors):
        work_queue.put(None)

    pool.close()
    pool.join()

    crystal_paths: List[Path] = destination.glob("**/*pkl")

    # Collate crystals and properties into list and dict, respectively
    crystals: List[ASUCrystal] = []
    property_dict = {}
    for path in crystal_paths:
        crystal, crystal_property_dict = load_object(path)  # loads pickle
        crystals.append(crystal)

        if len(property_dict) == 0:  # Fill property dict keys
            for property_name in crystal_property_dict.keys():
                property_dict[property_name] = []

        for property_name in property_dict.keys():
            property_dict[property_name].append(crystal_property_dict[property_name])

    # -- Save crystals as flattened numpy arrays in .npz. Save properties in
    #   pickled dataframe.
    pack_and_save(crystals, destination)  # saves crystal as flattened numpy array
    if len(property_dict) > 0:
        properties_dataframe = pd.DataFrame(data=property_dict)
        properties_dataframe.to_pickle(
            path=destination.with_name(destination.stem + "_properties").with_suffix(".pkl")
        )


def main(args):
    log: logging.Logger = setup_logger(__name__)

    num_processors: int = multiprocessing.cpu_count()
    log.info(f"Using {num_processors=}")

    if args.dataset == "mp20":
        if args.assume_p1_only:
            archive: Path = Path(DATA_DIRECTORY / "mp_20_assumeP1.tar.bz")
        else:
            archive: Path = Path(DATA_DIRECTORY / "mp_20.tar.bz")
    elif args.dataset == "mpts52":
        archive: Path = Path(DATA_DIRECTORY / "mpts_52.tar.bz")
    else:
        raise AttributeError
    extracted_data: Path = DATA_DIRECTORY / remove_suffix(archive)

    if extracted_data.exists() and extracted_data.is_dir():
        log.info(f"Found data pre-extracted at: {extracted_data.as_posix()}")
    else:
        try:
            extract(archive)
            log.info(f"Extracted archive")
        except FileNotFoundError as e:
            log.error(f"Could not find dataset archive at {archive.as_posix()}")
            raise e

    csvs: List[Path] = extracted_data.glob("**/*csv*")

    for csv_location in csvs:
        destination: Path = csv_location.with_suffix("")
        destination.mkdir(exist_ok=True)
        process_csv(log, args, csv_location, destination)


if __name__ == "__main__":
    # import tarfile
    # import os
    # file = tarfile.open('../data/mp_20_assumeP1.tar.bz', 'w:bz2')
    # path: Path = DATA_DIRECTORY / 'mp_20_assumeP1'
    # file.add(path.as_posix(), arcname=os.path.basename(path.as_posix()))
    # file.close()

    args = parser.parse_args()
    main(args)
