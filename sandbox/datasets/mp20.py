import os
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class MP20Dataset(Dataset):
    """MP-20 structures loaded from a CSV with inline CIF text.

    The reference CSV (models/adit/data/mp_20/raw/all.csv) stores each
    structure as CIF text in the `cif` column, keyed by `material_id` —
    there are no separate per-structure .cif files on disk, and (unlike the
    canonical CDVAE-style MP-20 distribution) no separate train/val/test
    CSVs ship with this checkout either (confirmed empirically: only
    raw/all.csv exists).

    So `split` is honored via a deterministic, reproducible partition of
    `all.csv` computed here, rather than read from disk: a fixed-seed random
    permutation of row indices is cut into train/val/test according to
    `train_frac`/`val_frac` (default 60/20/20 — matches the convention cited
    across the CDVAE/DiffCSP/WyFormer lineage in matgen_lit_2.xls). This is
    NOT guaranteed to reproduce the exact same row assignment as the
    original papers' splits (we don't know their seed or shuffling order),
    but it is internally consistent: the same seed always yields the same
    partition, so novelty comparisons against "train" are stable across runs
    and no longer leak "test" rows into the reference set.

    Previously `split` was silently ignored entirely — every call loaded an
    arbitrary `head(limit)` slice of the whole file regardless of the
    requested split. That bug is fixed here.
    """

    def __init__(self, root: str = "./data/mp20", split: str = "test", limit: Optional[int] = None,
                 seed: int = 42, train_frac: float = 0.6, val_frac: float = 0.2):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train'/'val'/'test', got {split!r}")
        self.root = root
        self.split = split
        self.seed = seed
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.structures = []
        self._load_data(limit=limit)

    def _split_assignment(self, n: int) -> np.ndarray:
        """Deterministic train/val/test label per row index, given
        (n, self.seed, self.train_frac, self.val_frac).
        """
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        n_train = min(round(n * self.train_frac), n)
        n_val = min(round(n * self.val_frac), n - n_train)
        labels = np.empty(n, dtype=object)
        labels[order[:n_train]] = "train"
        labels[order[n_train:n_train + n_val]] = "val"
        labels[order[n_train + n_val:]] = "test"
        return labels

    def _load_data(self, limit: Optional[int] = None):
        csv_path = os.path.join(self.root, "raw", "all.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        labels = self._split_assignment(len(df))
        df = df[labels == self.split]
        # limit applies AFTER split filtering — applying it before (as the
        # previous implementation effectively did, by ignoring split
        # entirely) would silently take an arbitrary slice of the whole
        # file instead of a slice of the requested split.
        if limit:
            df = df.head(limit)

        for _, row in df.iterrows():
            try:
                structure = CifParser.from_str(row["cif"]).parse_structures(primitive=True)[0]
                self.structures.append(structure)
            except Exception:
                continue

        print(f"Loaded {len(self.structures)} structures for {self.split} "
              f"(seed={self.seed}, train_frac={self.train_frac}, val_frac={self.val_frac})")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]


def build_mp20_dataloader(root: str = "./data/mp20", batch_size: int = 32, num_workers: int = 0,
                           split: str = "test") -> DataLoader:
    dataset = MP20Dataset(root, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
