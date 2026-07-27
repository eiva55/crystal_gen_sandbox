import datetime
from functools import lru_cache
import os
from pathlib import Path
import pickle
import random
import tarfile
from typing import Tuple

from omegaconf import OmegaConf

from aliases import *

SOURCE_DIRECTORY: Path = Path(__file__).parent.parent.absolute()
PROJECT_DIRECTORY: Path = SOURCE_DIRECTORY.parent.absolute()
DATA_DIRECTORY: Path = PROJECT_DIRECTORY / "data"
LOG_DIRECTORY: Path = PROJECT_DIRECTORY / "logs"
ASU_DICT_PATH: Path = PROJECT_DIRECTORY / "data/wyckoff_positions/clean_wyckoffs_in_asu_v6.json"
SHAPE_DECOMP_DICT_PATH: Path = PROJECT_DIRECTORY / "data/wyckoff_shape_decomposition.pkl"

auto_built_directories: Tuple[Path] = (
    DATA_DIRECTORY,
    LOG_DIRECTORY,
)

for path in auto_built_directories:
    path.mkdir(exist_ok=True)


def remove_suffix(path: Path) -> Path:
    while path.suffix:
        path = path.with_suffix("")
    return path


def extract(archive_path: Path) -> None:
    compression_standard: str = archive_path.suffix[1:]
    if compression_standard == "bz":
        compression_standard = "bz2"
    with tarfile.open(archive_path, f"r:{compression_standard}") as archive:
        archive.extractall(path=DATA_DIRECTORY)


def get_now_str() -> str:
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def get_random_hash(num_bits: int = 16) -> str:
    return str(random.getrandbits(num_bits))


@lru_cache
def get_source_dir() -> Path:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@lru_cache
def get_project_dir() -> Path:
    return os.path.dirname(get_source_dir())


def get_project_subdir(name: str) -> Path:
    abs_path: Path = os.path.join(get_project_dir(), name)
    if os.path.exists(abs_path) is False:
        os.mkdir(abs_path)
    return abs_path


def save_object(obj: Any, location: Path):
    location: Path = Path(location) if (not isinstance(location, Path)) else location
    if location.stem != ".pkl":
        location = location.with_suffix(".pkl")

    with open(location, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(location: Path) -> Any:
    with open(location, "rb") as handle:
        result = pickle.load(handle)
    return result


@lru_cache
def get_experiment_logs_dir() -> Path:
    return get_project_subdir("experiment_logs")


@lru_cache
def get_log_dir() -> Path:
    return get_project_subdir("logs")


def setup_experiment_directory(application: str, leaf_directory_modifier: str = "") -> Path:
    now: str = get_now_str()
    hash: str = get_random_hash()
    leaf_directory = now + f"_hash{hash}_" + leaf_directory_modifier
    directory: Path = os.path.join(get_experiment_logs_dir(), application, leaf_directory)
    if os.path.exists(os.path.join(get_experiment_logs_dir(), application)) is False:
        os.mkdir(os.path.join(get_experiment_logs_dir(), application))

    assert (
        os.path.exists(directory) is False
    ), f"experiment log directory: {directory} already exists..."
    os.mkdir(directory)
    return directory


def human_bytes_str(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB")
    power = 2**10

    for unit in units:
        if num_bytes < power:
            return f"{num_bytes:.1f} {unit}"

        num_bytes /= power

    return f"{num_bytes:.1f} TB"


def convert_wandb_config_to_hydra_config(wandb_config_path: str) -> DictConfig:
    config: DictConfig = OmegaConf.load(wandb_config_path)
    dict_config: dict = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(dict_config)
    return config
