import os
import pandas as pd
from pymatgen.io.cif import CifParser
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class MP20Dataset(Dataset):
    """MP-20 structures loaded from a CSV with inline CIF text.

    The reference CSV (models/adit/data/mp_20/raw/all.csv) stores each
    structure as CIF text in the `cif` column, keyed by `material_id` —
    there are no separate per-structure .cif files on disk.
    """

    def __init__(self, root: str = "./data/mp20", split: str = "test", limit: Optional[int] = None):
        self.root = root
        self.split = split
        self.structures = []
        self._load_data(limit=limit)

    def _load_data(self, limit: Optional[int] = None):
        csv_path = os.path.join(self.root, "raw", "all.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)

        for _, row in df.iterrows():
            try:
                structure = CifParser.from_str(row["cif"]).parse_structures(primitive=True)[0]
                self.structures.append(structure)
            except Exception:
                continue

        print(f"Loaded {len(self.structures)} structures for {self.split}")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]


def build_mp20_dataloader(root: str = "./data/mp20", batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    dataset = MP20Dataset(root, split="test")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
