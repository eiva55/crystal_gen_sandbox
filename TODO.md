# TODO

- [ ] WyFormer: `generate()` returns Wyckoff-position–encoded crystals
      (`{"group": ..., "sites": [...], "species": [...], "numIons": [...]}`),
      not explicit atomic coordinates. Reconstructing a real
      `pymatgen.Structure` requires expanding Wyckoff positions for the given
      space group — needs `pyxtal` (or direct `spglib` symmetry-operation
      tables), not a simple parser. Deferred; WyFormer currently participates
      in plain generation only, not in evaluate.py/metrics.
- [ ] Wire `cfg.dataset` (MP20Dataset) into evaluate.py for real
      novelty/uniqueness against the MP-20 reference set — blocked on
      confirming where real MP-20 data lives on disk.
