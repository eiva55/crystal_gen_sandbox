import os
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

class MP20Dataset(Dataset):
    def __init__(self, root: str = "./data/mp20", split: str = "test"):
        self.root = root
        self.split = split
        self.structures = []
        self._load_data()

    def _load_data(self):
        # Путь к CSV с CIF-файлами (как в ADiT)
        csv_path = os.path.join(self.root, "raw", "all.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Предположим, что в CSV есть колонка 'cif' или 'structure'
        # В MP20 обычно используют отдельные CIF-файлы, но упростим:
        # Будем читать CIF из папки raw/
        cif_dir = os.path.join(self.root, "raw")
        for idx, row in df.iterrows():
            cif_id = row.get('id', f"mp-{idx}")
            cif_file = os.path.join(cif_dir, f"{cif_id}.cif")
            if os.path.exists(cif_file):
                try:
                    parser = CifParser(cif_file)
                    structure = parser.get_structures()[0]
                    self.structures.append(structure)
                except:
                    continue
        print(f"Loaded {len(self.structures)} structures for {self.split}")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]

def build_mp20_dataloader(root: str = "./data/mp20", batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    dataset = MP20Dataset(root, split="test")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
