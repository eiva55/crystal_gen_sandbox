from unittest.mock import MagicMock, patch

from sandbox.contracts import BaseCrystalModel


class _DummyModel(BaseCrystalModel):
    def __init__(self, conda_env=None, ckpt_path=None):
        self.conda_env = conda_env
        self.ckpt_path = ckpt_path

    def generate(self, *a, **kw):
        return []

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path):
        pass


def test_flags_missing_checkpoint_path():
    model = _DummyModel(ckpt_path="/definitely/does/not/exist.ckpt")
    results = model.sanity_check()
    path_check = next(r for r in results if r.name == "path:ckpt_path")
    assert not path_check.ok


def test_passes_existing_checkpoint_path(tmp_path):
    ckpt = tmp_path / "fake.ckpt"
    ckpt.write_text("x")
    model = _DummyModel(ckpt_path=str(ckpt))
    results = model.sanity_check()
    path_check = next(r for r in results if r.name == "path:ckpt_path")
    assert path_check.ok


def test_checks_conda_env_availability():
    fake_proc = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("subprocess.run", return_value=fake_proc) as mock_run:
        model = _DummyModel(conda_env="some_env")
        results = model.sanity_check()
    assert mock_run.called
    env_check = next(r for r in results if r.name == "conda_env:some_env")
    assert env_check.ok


def test_no_checks_when_no_relevant_attributes():
    class _Bare(BaseCrystalModel):
        def generate(self, *a, **kw):
            return []

        def save_checkpoint(self, path):
            pass

        def load_checkpoint(self, path):
            pass

    assert _Bare().sanity_check() == []


def test_extra_checks_are_included_in_sanity_check():
    class _WithExtra(BaseCrystalModel):
        def generate(self, *a, **kw):
            return []

        def save_checkpoint(self, path):
            pass

        def load_checkpoint(self, path):
            pass

        def extra_checks(self):
            from sandbox.utils.check import CheckResult
            return [CheckResult(name="custom", ok=True, message="passed")]

    results = _WithExtra().sanity_check()
    assert any(r.name == "custom" for r in results)
