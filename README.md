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

Evaluate mode (generation + validity/uniqueness/novelty/S.U.N. against real MP-20 data):

```bash
python run.py model=crystaldit runner=evaluate runner.num_samples=10 runner.batch_size=10 dataset.limit=200
```

For a methodologically honest comparison across models (matching the scale
papers typically report at), use a large sample count and the full training
split as reference:

```bash
python run.py model=crystaldit runner=evaluate \
  runner.num_samples=1000 runner.batch_size=100 \
  dataset.limit=999999   # effectively "no limit" — takes the whole split
```

`dataset.limit=200` (the config default) is fine for quick smoke tests but
is far too small a reference set for novelty/S.U.N. to mean anything — with
a tiny reference, novelty inflates toward 1.0 almost regardless of model
quality (see "Metrics methodology" below for why).

Every run (generate or evaluate) sets `random_seed.seed` (default `42`, see
`configs/random_seed/basic.yaml`) as a global Python/NumPy/PyTorch seed at the
start of `run.py`. **This only reaches the orchestrator process** — each
model's real generation runs in its own conda env via `subprocess`, so the
seed only propagates as far as each model's own CLI/config actually supports:

| Model | Seed control | Mechanism |
|---|---|---|
| ADiT | Works | `seed=<value>` appended to its own Hydra-style CLI args |
| MiAD | Works | `seed:`, `num_samples:`, and `data.batch_size:` in `generate_miad_mp20.yaml` are all temporarily overwritten, then restored after the run (no CLI flags exist for any of the three). `data.batch_size` genuinely controls generation batching — confirmed by reading `lib/pipelines/ab_initio_generation.py`, not assumed — it does not affect the total sample count, only throughput. |
| WyFormer | No effect | no `--seed` flag; `generate()` prints a notice and ignores the override |
| SGEquiDiff | Fixed, not tunable | `generate_crystals.py` hardcodes `torch.manual_seed(0)` directly — always the same seed regardless of `random_seed.seed`, and doesn't read the `seed: 0` field already present in its own YAML configs |
| CrystalDiT | No effect | no seeding mechanism anywhere in the inference path (`generate_crystals.py`, `crystal_diffusion.py`, `diffusion/*.py`) — confirmed empirically: two runs with identical settings produced different compositions and atom counts |

So a seed sweep (`random_seed.seed=41,42,43`) gives genuinely different,
reproducible results per seed for ADiT and MiAD; for WyFormer and CrystalDiT
it just re-runs the same stochastic generation three times (the seed value
itself has no effect); for SGEquiDiff all three "seeds" produce the same
output, since the model ignores the request and always uses its own hardcoded
seed. Sweep several seeds in one command to see how much validity/uniqueness/novelty vary in practice:

```bash
python run.py --multirun runner=evaluate model=crystaldit \
  runner.num_samples=10 runner.batch_size=10 dataset.limit=200 \
  random_seed.seed=41,42,43
```

Each run's outputs land under Hydra's own output directory (`outputs/<date>/<time>/`
for a single run, `multirun/<date>/<time>/<job>/` per job for a sweep —
`runner.save_dir` resolves to `${hydra:runtime.output_dir}` so sweep jobs never
overwrite each other) and always include:

- `metrics.json` — the metrics dict returned by the runner (evaluate mode only)
- `run_summary.txt` — model name, seed, and a one-line result summary
- `tb/` — a TensorBoard event file with the same metrics logged as scalars
  (`metrics/<name>`, keyed by seed) plus the model/seed as text, so a seed
  sweep can be inspected visually instead of only via `metrics.json`:

```bash
tensorboard --logdir outputs      # single runs
tensorboard --logdir multirun     # sweeps
```

Generated CIFs are saved twice into `save_dir` under different prefixes:
`gen_*.cif` may already be written there by a model's own generation script
(e.g. CrystalDiT's `generate_crystals.py` defaults to writing directly into
its output dir with that prefix), and `evaluate.py` additionally re-saves
every structure as `eval_gen_*.cif` as a convenience for models whose own
scripts write elsewhere (ADiT, MiAD, SGEquiDiff). The `eval_gen_` prefix is
deliberately different from `gen_` — an earlier version used the same
prefix for both and silently clobbered most of CrystalDiT's own output due
to a 0- vs 1-indexing mismatch between the two save calls, leaving a stray
duplicate file behind. When loading CIFs back for offline metric
recomputation (see `sandbox/utils/load_structures.py`), prefer `eval_gen_*`
where it exists.

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
| ADiT | `ADiT` | 3.10ish | evaluate mode additionally passes `+trainer.limit_test_batches=1` — see "ADiT internal test-loop overhead" below |
| WyFormer | `WyFormer` | 3.12–3.13 (hard requirement) | needs `pyxtal` for structure reconstruction |
| MiAD | `miad` | 3.10 | no upstream requirements.txt — env exported from a working install |
| SGEquiDiff | `SGEquiDiff` | 3.10 | upstream pyproject.toml pins CUDA-specific torch; the exported env is the actual working CPU install. No batching speedup observed on CPU — generation time scales ~linearly with num_samples (~60s/sample), independent of batch_size. |
| CrystalDiT | `crystaldit` | 3.9 | `CUDA_VISIBLE_DEVICES=""` is forced in the subprocess env — its own `generate_crystals.py` auto-enables multi-GPU (`torch.cuda.device_count() > 1`) regardless of the `--device cpu` flag we pass, and its conda env ships a GPU-capable torch build (unlike the other four, which pin `+cpu` wheels), so this isolation isn't optional on a multi-GPU machine. |

`./setup_envs.sh` recreates all five from `envs/*.yml`. Env names are also
configurable per model (`model.conda_env=...`) if you name yours differently.

MP-20 reference data is expected at `models/adit/data/mp_20/raw/all.csv`
(inline CIF text per row) — this ships with the ADiT checkout; no separate
download needed as long as `models/adit/` is set up. There are no separate
train/val/test CSVs in this checkout (only `all.csv` exists) — see
`sandbox/datasets/mp20.py` below for how splits are derived from it.

### ADiT internal test-loop overhead

`eval_diffusion.py`'s `evaluate()` unconditionally calls
`trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)`
— a Lightning test loop over ADiT's *own* mp20/qm9/qmof150 test sets,
completely separate from (and not consumed by) this wrapper's
`diffusion_module.sampling.num_samples`. We never read its output
(`test_mp20/*`, `test_qm9/*`, `test_qmof150/*` metrics) — only the CIFs
ADiT's sampling step writes as a side effect, from
`models/adit/logs/eval_diffusion/runs/*/mp20_test_0`. The qmof150 branch
alone costs several minutes regardless of `num_samples` (observed
`test_qmof150/sampling_time` ~550s on a small run). `+trainer.limit_test_batches=0`
was tried first and silently broke generation entirely — the sampling
call turned out to live *inside* `test_step`, not to be an independent
step — so `=1` (one batch per dataset, not zero) is the actual fix: it cuts
the fixed cost from ~13 minutes to about a minute without breaking
generation. This is set unconditionally in `sandbox/models/adit.py`, not
exposed as a config knob, since there's no legitimate reason to want the
full internal test loop given we never consume its output.

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
5. Before trusting a config-driven CLI/YAML override you're adding for a
   new model, read the actual code path it's supposed to affect (as done
   for MiAD's `data.batch_size` above) rather than inferring its role from
   the field's name/location alone — an earlier version of this wrapper
   shipped an incorrect assumption about that exact field for a full
   development cycle before it was checked against the source.

## Tasks

- [x] Generation (`runner=default`): run one model, save CIFs + a plot.
- [x] Evaluation (`runner=evaluate`): generation + validity/uniqueness/novelty/S.U.N.
      metrics against real MP-20 reference structures.
- [x] Baseline (`model=random_baseline`): copies real structures verbatim, for
      sanity-checking that novelty isn't always trivially 1.0.
- [x] Sanity checks (`sanity_run=True`): fast wiring checks without real generation.
- [x] Reproducibility (partial, model-dependent): `random_seed.seed` is applied
      (Python/NumPy/PyTorch) at the start of every run, and supports Hydra
      multirun seed sweeps (`random_seed.seed=41,42,43`) — but the seed only
      actually reaches ADiT and MiAD (the other three models either ignore it
      or have no seeding mechanism at all). See the Quickstart table for the
      per-model breakdown.
- [x] Persisted results: `metrics.json` + `run_summary.txt` are written to
      each run's output dir (not just printed to stdout), and each sweep job
      gets its own output dir (`runner.save_dir=${hydra:runtime.output_dir}`)
      instead of overwriting a shared `./outputs`.
- [x] Config-driven metrics: `sandbox/metrics/crystal_metrics.py::CrystalMetrics`
      implements `BaseMetrics` and is instantiated from `configs/metrics/*.yaml`
      (`hydra.utils.instantiate(cfg.metrics)`) — swap or tune metrics from the
      command line (e.g. `metrics.compute_stability=false`) without touching
      `evaluate.py`.
- [x] Experiment tracking: metrics and run metadata are also logged to
      TensorBoard (`<save_dir>/tb/`).
- [x] Real validity metrics: structural (min interatomic distance > 0.5 Å,
      CDVAE-style threshold) and compositional (SMACT charge neutrality +
      Pauling electronegativity test) — replaces an earlier placeholder that
      always evaluated to ~1.0 regardless of sample size or structure
      quality. See "Metrics methodology" below.
- [x] Real train/val/test split: `MP20Dataset` now honors `split` via a
      deterministic seeded permutation of `all.csv` (no separate split CSVs
      ship with this checkout) — novelty is checked against `train` by
      default, matching the "does the model just memorize training data"
      convention used across the literature. Previously `split` was
      silently ignored entirely.
- [x] Gated S.U.N. pipeline: `un_rate` and `sun_rate_chgnet_approx` /
      `msun_rate_chgnet_approx` compute the paper-style joint metric (valid
      → unique-among-valid → novel-among-unique-valid → stable-among-UN),
      alongside (not replacing) the older independent validity/uniqueness/
      novelty rates, which remain useful diagnostics on their own axis.
- [x] CHGNet-relaxed stability: `sandbox/metrics/stability.py` now relaxes
      structures (ASE FIRE, up to 100 steps, `FrechetCellFilter`) via
      `chgnet.model.StructOptimizer` before computing `e_above_hull`,
      instead of a single-point energy on the as-generated structure. The
      reference convex hull cache is versioned (`REFERENCE_CACHE_VERSION`)
      so a cache built under the old single-point methodology is refused
      rather than silently reused as if it were relaxed.
- [ ] Real DFT-based S.U.N. (VASP): still not implemented, and not
      realistically feasible in this environment (no VASP license,
      pseudopotentials, or the CPU-hours real DFT relaxation would need).
      CHGNet-relaxed S.U.N. is a much closer approximation than the earlier
      single-point version, but is still expected to diverge from
      paper-reported numbers — see "Known discrepancies" below.
- [ ] WyFormer's own MLFF relaxation step (CHGNet/MACE/ORB, via its
      `cryspr/` scripts) is still skipped — we only do symmetry-consistent
      structural reconstruction via `pyxtal`, not the energy-relaxation
      pass WyFormer's own pipeline performs as part of generation itself.
      This is a *different* relaxation from the evaluation-time CHGNet
      relaxation above (that one runs on whatever structure any model
      output, after the fact, purely for the stability metric) — this TODO
      is about WyFormer's own generation-time step, still open.
- [ ] Wire `configs/dataset` / `configs/task` more generally — right now
      `evaluate.py` is the only place that consumes the dataset.
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
- `evaluate_generation(model, num_samples, batch_size, device, save_dir=None, dataset=None, task=None, viz_enabled=False, metrics=None, **kwargs) -> Dict[str, float]`
  — if `metrics` (a `BaseMetrics` instance from `cfg.metrics`) is passed, it's
  used via `metrics.compute(structures, reference)`; otherwise falls back to
  the hardcoded `CrystalMetrics.compute_all(...)` path for callers that don't
  wire `cfg.metrics`.

## Dataset (`sandbox/datasets/mp20.py`)

`MP20Dataset(root, split="test", limit=None, seed=42, train_frac=0.6, val_frac=0.2)`

Since no separate train/val/test CSVs ship with this checkout (only
`raw/all.csv`), `split` is honored via a deterministic seeded permutation of
row indices computed at load time (`numpy.random.default_rng(seed)`), cut
into train/val/test per `train_frac`/`val_frac` (default 60/20/20). This is
**not** guaranteed to reproduce the exact row assignment of the original
papers' splits (their seed/shuffling order is unknown to us), but it is
internally consistent — the same seed always yields the same partition, so
novelty checks against `train` are stable across runs and don't leak `test`
rows into the reference set. `limit` is applied *after* split filtering, not
before (an earlier version effectively ignored `split` entirely and applied
`limit` to the whole file in whatever order it happened to be stored).

`configs/dataset/mp20.yaml` defaults to `split: train` (not `test`) —
novelty is a memorization check, so it should compare against what the
model could plausibly have seen during training, not a held-out set.
`seed` is wired to `${random_seed.seed}` by default, so it moves together
with the rest of the run's reproducibility settings.

For building the CHGNet stability reference (a separate concern from
novelty — the convex hull just needs broad coverage of known
thermodynamically-relevant compositions, not a train/test-clean set), the
reference-cache rebuild script pulls all three splits combined (~45k
structures), not just `train`.

## Metrics methodology (`sandbox/metrics/crystal_metrics.py`, `stability.py`)

`CrystalMetrics` implements the `BaseMetrics` contract and is instantiable
from config (`configs/metrics/basic.yaml`):

```yaml
_target_: sandbox.metrics.crystal_metrics.CrystalMetrics
stability_reference_path: cache/chgnet_full_reference.json
compute_stability: true
relax_fmax: 0.1
relax_steps: 100
```

### Validity

- `compute_structural_validity(structure)` — all pairwise interatomic
  distances (via pymatgen's periodic `distance_matrix`, not raw Cartesian)
  must exceed 0.5 Å.
- `compute_compositional_validity(structure)` — `smact.screening.smact_validity`
  with library defaults (`use_pauling_test=True`, `include_alloys=True`):
  charge neutrality + Pauling electronegativity test.
- `compute_validity(structures)` — structural AND compositional, reported
  as a fraction of the whole generated set. Both sub-rates
  (`structural_validity`, `compositional_validity`) are also reported
  separately in `compute_all`'s output, matching how the literature usually
  breaks this down.

This replaces an earlier placeholder (`s is not None and len(s) > 0`) that
evaluated to ~1.0 regardless of N or actual structure quality — that number
was measuring "did generation not crash," not validity in any sense
comparable to what papers report.

### Uniqueness / Novelty — independent vs. gated

`compute_uniqueness(structures)` and `compute_novelty(structures, reference)`
each still operate independently over the *whole* generated set (matching
their original, pre-refactor behavior) — useful diagnostics, but not what
papers report as "S.U.N. rate." Both StructureMatcher-based
(`ltol=0.2, stol=0.3, angle_tol=5`) comparisons are grouped by
`reduced_formula` first (`_group_by_formula`) purely for performance —
`StructureMatcher.fit()` already rejects mismatched compositions
internally, so grouping doesn't change any result, it just skips calling
`fit()` on pairs that can never match, which matters once the reference set
is in the thousands rather than hundreds.

`compute_all` additionally builds the **gated** chain that matches the
paper convention — each stage filters the *previous* stage's survivors,
not the whole generated set independently:`un_rate` is the resulting fraction of *all* generated structures that are
simultaneously valid+unique+novel — this, not the independent `uniqueness`/
`novelty` numbers above, is the one comparable to a paper's reported
"UN rate."

### Stability (CHGNet-relaxed, still not DFT)

`stability.py` relaxes each structure (`chgnet.model.StructOptimizer`, ASE
FIRE, `FrechetCellFilter`, cell relaxation on, up to `relax_steps` — a hard
cap, not a convergence guarantee) before computing `e_above_hull` against a
per-structure chemical-subsystem `PhaseDiagram` built from cached reference
entries. The reference entries are relaxed the same way and cached in
`cache/chgnet_full_reference.json`, tagged with `REFERENCE_CACHE_VERSION` —
a cache built under the old single-point (unrelaxed) methodology is
detected and refused rather than silently reused (comparing a relaxed
generated structure against an unrelaxed reference hull would be
meaningless). Rebuild via `sandbox.metrics.stability.build_reference_entries`.

`compute_all` calls `compute_stability` twice — once over the whole
generated set (ungated, `stable_rate_chgnet_approx` / `metastable_rate_chgnet_approx`,
unchanged in meaning from before this refactor), and once over the UN-gated
subset only (`stable_among_un_rate` / `metastable_among_un_rate`, which
combine with `un_rate` into `sun_rate_chgnet_approx` / `msun_rate_chgnet_approx`
— the paper-style joint metric). The second call reuses an `energy_cache`
dict (keyed by `id(structure)`) populated by the first call — since the
UN subset is a subset of the *same Python objects*, not copies, every
structure in it is a guaranteed cache hit, so the "gated" stability pass
costs nothing extra in practice.

This is still a CHGNet approximation, not DFT — `stable_threshold=0.0`
(strictly on the hull) is essentially never hit by any model, CHGNet or
DFT alike; `metastable_threshold=0.1` eV/atom is the more informative
number for cross-model comparison.

### Chemical validity gap — resolved (SMACT calibration)

`compute_compositional_validity` originally used `smact_validity`'s library
defaults (`consensus=3, commonality="medium"`), which filter out oxidation
states below a literature-frequency threshold — a newer addition to the
`smact` library that predates the CDVAE-lineage methodology these papers
use (effectively no frequency filtering: any historically observed
oxidation state counts). This produced a systematic ~24-31 п.п. gap below
authors' reported numbers on every model checked (ADiT, CrystalDiT,
SGEquiDiff).

Confirmed on ADiT's N=1000 set: default settings gave 64.5% vs. the
paper's 90.83% (Δ≈-26 п.п.); `consensus=1, commonality="low"` gives 92.9%
(Δ≈+2 п.п.). `use_pauling_test=False` barely moved the number in the same
experiment (64.5% → 64.7%), ruling out the Pauling electronegativity test
as the driver. `compute_compositional_validity` now uses `consensus=1,
commonality="low"` by default. Post-fix validity deltas: ADiT +1.5 п.п.,
CrystalDiT -0.2 п.п., SGEquiDiff +7.0 п.п. — MiAD and WyFormer have no
comparable authors' chemical-validity number to check against.

### Stability comparison bug: gated vs. ungated mismatch (fixed)

The first version of the cross-paper stability comparison compared our
ungated `metastable_rate_chgnet_approx` (computed over the WHOLE generated
set) against authors' S.U.N./S.S.U.N. numbers (which are gated — Stable
AND Unique AND Novel jointly) for four of five models — comparing a
broader metric to a narrower one, which inflated the apparent gap
substantially (e.g. MiAD showed +84.0 п.п. on the wrong basis). Recomputed
against `msun_rate_chgnet_approx` (our own gated Metastable+Unique+Novel
joint metric — the actual apples-to-apples counterpart), the gaps roughly
halve: MiAD +49.9 п.п., SGEquiDiff +32.1 п.п., CrystalDiT +47.3 п.п.,
WyFormer -7.6 п.п. (previously showed -0.7 п.п. — that apparent near-match
was itself an artifact of comparing mismatched metrics, not a real close
fit). ADiT's authors' number is already reported as a plain (ungated)
"Metastable" rate, so its original ungated-vs-ungated comparison
(+15.4 п.п.) needed no correction. Recomputing the gated numbers required
re-relaxing the (now larger, post-SMACT-fix) UN-gated subset with CHGNet,
not just redoing arithmetic on cached values — the UN subset's size itself
changed once compositional validity stopped over-filtering.

### Remaining gap: CHGNet approximation vs. real DFT

Even with both fixes above, all five models' gated stability numbers still
run above the authors' DFT-based S.U.N./S.S.U.N. figures (+15 to +50 п.п.
depending on model). Real DFT-based S.U.N. is not implemented and not
realistically feasible in this environment (no VASP license,
pseudopotentials, or the CPU-hours real DFT relaxation would need) — the
comparison table's stability-delta column notes explicitly that the two
sides are different metrics for this reason, not a claim that our models
are meaningfully more "stable" than the literature reports. Likely
contributors: CHGNet has known systematic energy biases relative to DFT
that vary by chemistry; our reference convex hull is built from a
CHGNet-relaxed MP-20 subset (~45k structures), not the (likely larger,
DFT-computed) reference the original papers use; `relax_steps=100` is a
hard cap, not a convergence guarantee; `stable_threshold=0.0` is
essentially never satisfied by any model under either methodology, so all
signal concentrates in the `metastable_threshold=0.1` eV/atom band.

Our own gated M.S.U.N. numbers, for reference: ADiT 60.3%, CrystalDiT
56.9%, MiAD 58.1%, SGEquiDiff 44.6%, WyFormer 27.6% — the diffusion-style
models (ADiT/CrystalDiT/MiAD) cluster together noticeably above
SGEquiDiff, with WyFormer lowest. Not yet interpreted architecturally.

### Operational incident: concurrent MiAD runs can corrupt the config

`MiADModel.generate()` temporarily overwrites
`generate_miad_mp20.yaml` and restores it in a `finally` block — this is
not safe against two `generate()` calls running at the same time: in
practice, two runs launched close together both overwrote the same file,
and one's restore raced the other's read, leaving the file stuck on an
overridden value with no error anywhere (silent, not a crash). A file lock
(`generate_miad_mp20.yaml.lock`, `os.O_CREAT | os.O_EXCL`) now makes a
second concurrent call fail loudly instead. This is a `MiADModel`-specific
fix (it's the only model that mutates external file state as its
override mechanism) — the other four models pass overrides via CLI args,
which don't have this class of hazard. Don't run more than one `model=miad`
evaluate/generate at a time regardless; the lock is a safety net, not a
substitute for that.

Toggle or point metrics at a different reference from the command line, e.g.:

```bash
python run.py model=crystaldit runner=evaluate metrics.compute_stability=false
```
