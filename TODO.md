# TODO

## Done

- [x] WyFormer structural reconstruction: pyxtal.from_random() now converts
      Wyckoff-gene JSON into real pymatgen.Structure objects (see
      sandbox/models/wyformer_reconstruct.py), reusing the same approach as
      WyFormer's own cryspr/relaxer.py::single_pyxtal().
- [x] Wire `cfg.dataset` (MP20Dataset) into evaluate.py for real
      novelty/uniqueness against the MP-20 reference set — resolved
      (models/adit/data/mp_20).
- [x] MP20Dataset `split` bug: previously silently ignored — every call
      loaded an arbitrary `head(limit)` slice of the whole file regardless
      of the requested split. Now honored via a deterministic seeded
      permutation (60/20/20 default), `limit` applied after split
      filtering. `configs/dataset/mp20.yaml` default changed to
      `split: train` (novelty is a memorization check).
- [x] Placeholder validity metric (`s is not None and len(s) > 0`, always
      ~1.0 regardless of N) replaced with real structural (min interatomic
      distance > 0.5 Å) + compositional (SMACT charge neutrality) validity.
- [x] Uniqueness/novelty restructured into a gated pipeline
      (valid → unique-among-valid → novel-among-unique-valid vs. train) —
      `un_rate` and `sun_rate_chgnet_approx` / `msun_rate_chgnet_approx`
      now match the paper convention (a joint metric, not three
      independent percentages). Old independent rates kept as diagnostics.
- [x] CHGNet single-point energy replaced with CHGNet-driven relaxation
      (ASE FIRE, up to 100 steps) before computing e_above_hull — much
      closer to the CHGNet/M3GNet/ORB-relaxation convention the literature
      uses, though still not DFT. Reference cache versioned
      (`REFERENCE_CACHE_VERSION`) so a pre-relaxation cache is refused, not
      silently reused.
- [x] `crystal_metrics.py` performance: uniqueness/novelty comparisons
      grouped by `reduced_formula` before calling `StructureMatcher.fit()`
      — same results, avoids O(n×m) calls that can never match once the
      reference set is in the thousands.
- [x] CrystalDiT missing `CUDA_VISIBLE_DEVICES=""` — its own
      `generate_crystals.py` auto-enables multi-GPU regardless of our
      `--device cpu` flag, and unlike the other four models its conda env
      ships a GPU-capable torch build. Fixed by forcing the env var in the
      subprocess call, matching miad.py/sgequidiff.py.
- [x] ADiT unconditional internal `trainer.test()` loop (qm9/qmof150,
      ~13 min fixed cost, output never consumed) — `+trainer.limit_test_batches=0`
      broke generation entirely (the sampling call lives inside
      `test_step`); `=1` is the actual fix, cuts cost to ~1 minute.
- [x] `evaluate.py` re-save filename collision: re-saving generated
      structures as `gen_{i}.cif` collided with CrystalDiT's own
      `generate_crystals.py` (which defaults to the same prefix, writing
      directly into the same `save_dir`) due to a 0- vs 1-indexing
      mismatch — silently overwrote most of CrystalDiT's own output,
      leaving one duplicate file behind. Fixed by using `eval_gen_`
      instead of `gen_` for the wrapper's re-save.
- [x] MiAD `num_samples`/`batch_size` weren't wired to anything — hardcoded
      to whatever `models/miad/saved_configs/generate_miad_mp20.yaml`
      happened to contain (N=20). Now both are temporarily overridden in
      that YAML before each run and restored after, same mechanism already
      used for `seed`. `data.batch_size` was initially assumed (incorrectly)
      to only control reference-data loading — reading
      `lib/pipelines/ab_initio_generation.py` showed it directly determines
      the generation batch size (`total_batch_size = min(config.num_samples,
      data.config.batch_size)`); doesn't affect total sample count, only
      throughput, but is now wired correctly rather than documented wrong.
- [x] MiAD full N=1000 evaluate run completed — all five models now have
      comparable N=1000, full-train-split-reference results in
      `crystal_models_comparison_v2_template.xlsx`.
- [x] Concurrent-run hazard in `MiADModel.generate()`: two overlapping
      calls raced on the shared YAML override/restore, corrupting the real
      config file with no error raised (discovered when two runs were
      accidentally launched close together). Fixed with an exclusive file
      lock (`generate_miad_mp20.yaml.lock`) that makes a second concurrent
      call fail loudly instead of silently corrupting state.
- [x] Chemical validity gap root-caused and fixed: `smact_validity`'s
      library defaults (`consensus=3, commonality="medium"`) filter
      oxidation states by literature frequency, which the CDVAE-lineage
      papers' methodology doesn't do. Switched to `consensus=1,
      commonality="low"` — confirmed on ADiT (64.5%→92.9% vs. paper's
      90.83%); `use_pauling_test` ruled out as a contributing factor
      (barely moved the number). Post-fix deltas: ADiT +1.5 п.п.,
      CrystalDiT -0.2 п.п., SGEquiDiff +7.0 п.п.
- [x] Stability comparison bug fixed: cross-paper stability deltas were
      comparing our ungated `metastable_rate_chgnet_approx` against
      authors' gated S.U.N./S.S.U.N. numbers for four of five models
      (comparing a broader metric to a narrower one, inflating the
      apparent gap — e.g. MiAD showed +84.0 п.п. on the wrong basis).
      Fixed to compare against `msun_rate_chgnet_approx` (our own gated
      counterpart) instead — gaps roughly halve. Required re-relaxing the
      UN-gated subset with CHGNet (it grew once the SMACT fix above
      stopped over-filtering), not just recomputing arithmetic on cached
      numbers.

## Open

- [ ] Diffusion-style models (ADiT M.S.U.N. 60.3%, CrystalDiT 56.9%, MiAD
      58.1%) cluster noticeably above SGEquiDiff (44.6%) and WyFormer
      (27.6%) on our own gated stability metric. Not investigated — could
      be architecture-related, could be a relaxation-convergence artifact
      (structures needing more than the 100-step relax cap), unknown.
- [ ] Remaining +15 to +50 п.п. gap between our gated M.S.U.N. and authors'
      DFT-based S.U.N./S.S.U.N., after both the SMACT and gated-vs-ungated
      fixes below — expected and explicitly labeled in the comparison
      table as a methodology difference (CHGNet-approx vs. real DFT), not
      further investigated since closing it would require actual DFT,
      which isn't feasible here (see the DFT item further down).
- [ ] Real DFT-based S.U.N. (VASP): not implemented, not realistically
      feasible in this environment. CHGNet-relaxed is the practical ceiling
      here — flag any comparison against paper-reported DFT S.U.N. numbers
      as methodology-different, not model-quality-different, until/unless
      real DFT becomes available.
- [ ] WyFormer's own MLFF relaxation step (cryspr/ scripts: CHGNet/MACE/ORB)
      is still skipped in generation — we only do pyxtal structural
      reconstruction. This is separate from the evaluation-time CHGNet
      relaxation in stability.py (which runs post-hoc on whatever any
      model output, regardless of model) — WyFormer's own paper applies
      relaxation as part of producing its final structures in the first
      place, which we don't reproduce.
- [ ] Clarify what `wyformer-generate --firm-n-samples N` actually
      guarantees — empirically N=1000 requested produced 999 reconstructed
      structures (off by one, not exact).
- [ ] CrystalDiT non-reproducibility: confirmed empirically (two runs, same
      settings, different compositions/atom counts) and by code inspection
      — no `torch.manual_seed`/equivalent exists anywhere in the actual
      inference path. Gap in upstream model code, not in our wrapper.
- [ ] SGEquiDiff seed is hardcoded (`torch.manual_seed(0)`, literal, not
      read from its own `seed: 0` config field) — every run uses the same
      fixed seed regardless of our `random_seed.seed` override. Not
      user-tunable without patching SGEquiDiff's own script.
- [ ] SGEquiDiff generation time scales ~linearly with `num_samples`
      independent of `batch_size` on CPU (~60s/sample either way, no
      batching speedup observed) — N=1000 takes ~16.5h. No fix identified;
      just a hard constraint on how large a comparable run can be for this
      model without GPU access.
- [ ] Wire `configs/dataset` / `configs/task` more generally — right now
      `evaluate.py` is the only place that consumes the dataset.
