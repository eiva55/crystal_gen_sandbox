# crystal_gen_sandbox

## Intro

A unified Hydra-based wrapper around five external crystal-structure
generative models (ADiT, WyFormer, MiAD, SGEquiDiff, CrystalDiT). Each model
lives in its own conda environment (their dependencies conflict with each
other), and is invoked as a subprocess — the wrapper gives them a single
contract (`generate() -> List[pymatgen.Structure]`), a single entrypoint
(`run.py`), and shared metrics/baselines/visualization on top.

## Quickstart

```bash
./setup_envs.sh          # creates the 5 conda envs (ADiT, WyFormer, miad, SGEquiDiff, crystaldit)
                         # from envs/*.yml — see "Environments" below
pip install -r requirements.txt --break-system-packages
./test.sh                # fast unit tests, no conda envs / real generation needed
python run.py model=adit sanity_run=True   # verify wiring (paths, conda env) in seconds
python run.py model=adit runner.num_samples=5 runner.batch_size=5
```

Model overrides:

```bash
python run.py model=adit ...
python run.py model=wyformer ...
python run.py model=miad ...
python run.py model=sgequidiff ...
python run.py model=crystaldit ...
python run.py model=random_baseline ...   # copies real MP-20 structures verbatim — for sanity-checking metrics, not a real generator
```

Evaluate mode (generation + validity/uniqueness/novelty against real MP-20 data):

```bash
python run.py model=crystaldit runner=evaluate runner.num_samples=10 runner.batch_size=10 dataset.limit=200
```

Visualization (element-distribution plot, saved alongside the generated CIFs) is on by
default whenever `runner.save_dir` is set; disable with `viz.enabled=false`.

Sanity check any model without running real generation:

```bash
python run.py model=<name> sanity_run=True
```

This checks conda-env availability and declared checkpoint/data paths in
seconds, instead of waiting through a real run (ADiT alone takes ~4 minutes
per call). Models can add extra checks via `extra_checks()` — see
`sandbox/models/wyformer.py` for an example (checks `pyxtal` is importable).

## Environments

Each model runs in its own conda environment, exported from a known-working
CPU-only setup:

| Model | conda env name | Python | Notes |
|---|---|---|---|
| ADiT | `ADiT` | 3.10ish | |
| WyFormer | `WyFormer` | 3.12–3.13 (hard requirement) | needs `pyxtal` for structure reconstruction |
| MiAD | `miad` | 3.10 | no upstream requirements.txt — env exported from a working install |
| SGEquiDiff | `SGEquiDiff` | 3.10 | upstream pyproject.toml pins CUDA-specific torch; the exported env is the actual working CPU install |
| CrystalDiT | `crystaldit` | 3.9 | |

`./setup_envs.sh` recreates all five from `envs/*.yml`. Env names are also
configurable per model (`model.conda_env=...`) if you name yours differently.

MP-20 reference data is expected at `models/adit/data/mp_20/raw/all.csv`
(inline CIF text per row) — this ships with the ADiT checkout; no separate
download needed as long as `models/adit/` is set up.

## Adding a new model

1. Create `sandbox/models/<name>.py`. The class must implement
   `generate(num_samples, batch_size, device, save_dir=None, **kwargs) ->
   List[pymatgen.Structure]`, `save_checkpoint(path)`, `load_checkpoint(path)`
   (see `sandbox/contracts/base.py` for the full `BaseCrystalModel` contract).
   Optionally override `extra_checks()` for anything `sanity_run` can't infer
   generically (e.g. a package that must be importable inside the model's
   own conda env).
2. If the model's own script writes CIF/POSCAR files rather than returning
   structures directly, parse them back with
   `sandbox.utils.load_structures.load_structure_files(dir, pattern)` at the
   end of `generate()` — see `sandbox/models/miad.py` for the simplest
   example, `sandbox/models/wyformer.py` for a two-step (generate +
   reconstruct) example.
3. Add `configs/model/<name>.yaml` with `_target_: sandbox.models.<name>.<ClassName>`.
4. Export its conda env to `envs/<name>.yml` (`conda env export -n <env> --no-builds`)
   and add it to `setup_envs.sh`.

## Tasks

- [x] Generation (`runner=default`): run one model, save CIFs + a plot.
- [x] Evaluation (`runner=evaluate`): generation + validity/uniqueness/novelty
      metrics against real MP-20 reference structures.
- [x] Baseline (`model=random_baseline`): copies real structures verbatim, for
      sanity-checking that novelty isn't always trivially 1.0.
- [x] Sanity checks (`sanity_run=True`): fast wiring checks without real generation.
- [ ] WyFormer energy relaxation (CHGNet/MACE/ORB) — currently only
      symmetry-consistent structural reconstruction via `pyxtal` is done, not
      the energy relaxation step WyFormer's own `cryspr/` scripts perform.
- [ ] Wire `configs/dataset` / `configs/task` more generally — right now
      `evaluate.py` is the only place that consumes the dataset; other
      metrics (`BaseMetrics` subclasses beyond `CrystalMetrics`) aren't
      pluggable via config yet.
- See `TODO.md` for the current, evolving list.

# Contracts

## BaseCrystalModel (`sandbox/contracts/base.py`)

- `generate(num_samples, batch_size, device, save_dir=None, **kwargs) -> List[pymatgen.Structure]`
- `save_checkpoint(path) -> None` / `load_checkpoint(path) -> None`
  (no-ops for these five models — they wrap pretrained external checkpoints,
  not something we train ourselves)
- `to(device) -> self`
- `sanity_check() -> List[CheckResult]` — generic, inspects `conda_env` /
  `*_path` / `*_dir` attributes automatically; not meant to be overridden
  directly (see `extra_checks()` instead)
- `extra_checks() -> List[CheckResult]` — override per-model for anything the
  generic scan can't infer

## BaseTask (`sandbox/tasks/generation.py`)

- `run(model, num_samples, batch_size, device, save_dir=None, **kwargs) -> List[Structure]`
- `visualize(structures, outdir) -> Optional[str]` — saves an element-distribution PNG

## Runners (`sandbox/runners/`)

- `run_generation(task, model, num_samples, batch_size, device, save_dir, viz_enabled=False, **kwargs) -> List[Structure]`
- `evaluate_generation(model, num_samples, batch_size, device, save_dir=None, dataset=None, task=None, viz_enabled=False, **kwargs) -> Dict[str, float]`

## Metrics (`sandbox/metrics/crystal_metrics.py`)

- `compute_validity(structures) -> float`
- `compute_uniqueness(structures) -> float`
- `compute_novelty(structures, reference) -> float`
- `compute_all(structures, reference=None) -> Dict[str, float]`
