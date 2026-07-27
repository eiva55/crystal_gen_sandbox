"""Regression tests protecting the Hydra config <-> model wiring.

These do NOT launch real subprocesses (that needs each model's own conda
env, checkpoints, and minutes of runtime) — they only check that each
configs/model/*.yaml resolves to a real class conforming to the contract,
and that generate() shells out with the right conda env / cwd.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

from sandbox.contracts import BaseCrystalModel

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "model"


def _model_config_files():
    return sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", _model_config_files(), ids=lambda p: p.stem)
def test_model_target_resolves_and_conforms_to_contract(config_path):
    cfg = OmegaConf.load(config_path)
    model_cls = get_class(cfg._target_)
    assert issubclass(model_cls, BaseCrystalModel)


@pytest.mark.parametrize(
    "module_name,class_name,expected_marker",
    [
        ("sandbox.models.adit", "ADiTModel", "ADiT"),
        ("sandbox.models.wyformer", "WyFormerModel", "WyFormer"),
        ("sandbox.models.miad", "MiADModel", "miad"),
        ("sandbox.models.sgequidiff", "SGEquiDiffModel", "SGEquiDiff"),
        ("sandbox.models.crystaldit", "CrystalDiTModel", "crystaldit"),
    ],
)
def test_generate_invokes_subprocess_with_right_env(module_name, class_name, expected_marker, tmp_path):
    import importlib

    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    model = model_cls()

    fake_process = MagicMock()
    fake_process.wait.return_value = None
    fake_process.returncode = 0

    with patch("subprocess.Popen", return_value=fake_process) as mock_popen, \
         patch("sandbox.utils.load_structures.load_structure_files", return_value=[]):
        model.generate(num_samples=1, batch_size=1, device="cpu", save_dir=str(tmp_path))

    assert mock_popen.called
    call = mock_popen.call_args
    cmd = call.args[0] if call.args else call.kwargs.get("args")
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    assert expected_marker in cmd_str
