"""Standalone converter: WyFormer's Wyckoff-gene JSON -> CIF files via pyxtal.

Runs inside the WyFormer conda env (which has pyxtal as a hard dependency,
per its pyproject.toml) — this file has no dependency on the rest of this
repo so it can be invoked as a plain subprocess, the same way the other
four models' own generation scripts are invoked.

This does structural reconstruction only (real, symmetry-consistent atomic
coordinates) — it does NOT run the MLFF relaxation step (CHGNet/MACE/ORB)
that WyFormer's own cryspr/ scripts perform. See TODO.md.
"""
import argparse
import gzip
import json
import os
from pathlib import Path

from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix


def convert_one(gene: dict, out_path: Path, max_count: int = 20) -> None:
    tm = Tol_matrix(prototype="atomic", factor=1.3)
    candidate = pyxtal()
    candidate.from_random(
        dim=3,
        group=gene["group"],
        species=gene["species"],
        numIons=gene["numIons"],
        sites=gene["sites"],
        tm=tm,
        max_count=max_count,
    )
    candidate.to_file(str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with gzip.open(args.input, "rt") as f:
        genes = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    ok, failed = 0, 0
    for i, gene in enumerate(genes):
        out_path = Path(args.output_dir) / f"wyformer_{i}.cif"
        try:
            convert_one(gene, out_path)
            ok += 1
        except Exception as exc:
            print(f"Skipping gene {i} (group={gene.get('group')}): {exc}")
            failed += 1
    print(f"Reconstructed {ok} structures, skipped {failed}.")


if __name__ == "__main__":
    main()
