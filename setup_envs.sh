#!/usr/bin/env bash
set -euo pipefail

# Recreates the five isolated conda environments this project shells out to.
# These were exported (`conda env export --no-builds`) from a working
# CPU-only setup — the declared manifests in models/*/pyproject.toml or
# requirements.txt may ask for different (often GPU/CUDA-specific) versions
# than what's actually installed and confirmed working here.
# WyFormer specifically requires Python 3.12-3.13 (see models/wyformer/pyproject.toml);
# the other four use Python 3.9-3.10 — do not try to unify them into one env.

conda env create -f envs/adit.yml
conda env create -f envs/wyformer.yml
conda env create -f envs/miad.yml
conda env create -f envs/sgequidiff.yml
conda env create -f envs/crystaldit.yml
