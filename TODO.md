# TODO

- [x] WyFormer structural reconstruction: pyxtal.from_random() now converts
      Wyckoff-gene JSON into real pymatgen.Structure objects (see
      sandbox/models/wyformer_reconstruct.py), reusing the same approach as
      WyFormer's own cryspr/relaxer.py::single_pyxtal().
- [ ] WyFormer energy relaxation: WyFormer's own cryspr scripts
      (cryspr_pyxtal_chgnet.py / _mace.py / _orb.py) additionally relax the
      pyxtal-reconstructed structure with an ML force field (CHGNet/MACE/ORB)
      before reporting it as final. We currently skip this — reconstructed
      structures are symmetry-consistent but not energy-relaxed. Revisit if
      WyFormer's validity/novelty numbers look off compared to the other
      four models.
- [ ] Clarify what `wyformer-generate --firm-n-samples N` actually guarantees
      — a request for num_samples=2 produced 10 reconstructed structures.
- [ ] Wire `cfg.dataset` (MP20Dataset) into evaluate.py for real
      novelty/uniqueness against the MP-20 reference set — blocked on
      confirming where real MP-20 data lives on disk (now resolved:
      models/adit/data/mp_20).
- [ ] CrystalDiT non-reproducibility: confirmed empirically (two runs, same
      settings, different compositions/atom counts) and by code inspection —
      no `torch.manual_seed`/equivalent exists anywhere in the actual
      inference path (`generate_crystals.py`'s CPU branch
      `generate_crystals_single_device`, `crystal_diffusion.py`,
      `diffusion/models.py`, `diffusion/gaussian_diffusion.py`). This is a gap
      in the upstream model code, not in our wrapper — our `seed=` override
      correctly has no effect because there's nothing to pass it to.
- [ ] SGEquiDiff seed is hardcoded (`torch.manual_seed(0)`, literal, not read
      from its own `seed: 0` config field) — every run uses the same fixed
      seed regardless of our `random_seed.seed` override. Not user-tunable
      without patching SGEquiDiff's own script.
