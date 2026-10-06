"""Загрузка MP-20 специально для окружения equivariance_ablation, где
установлена pymatgen==2023.10.11 (метод CifParser.get_structures,
не parse_structures, как в остальных окружениях репозитория). Не переиспользует
sandbox.datasets.mp20.MP20Dataset напрямую, чтобы не завязывать общий, уже
используемый другими частями репозитория модуль на версию pymatgen
конкретно этого изолированного окружения.
"""
import os
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser


def load_mp20_structures(root, split, limit=None, seed=42, train_frac=0.6, val_frac=0.2):
    csv_path = os.path.join(root, "raw", "all.csv")
    df = pd.read_csv(csv_path)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_train = round(len(df) * train_frac)
    n_val = round(len(df) * val_frac)
    labels = np.empty(len(df), dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"

    df = df[labels == split]
    if limit:
        df = df.head(limit)

    structures = []
    for _, row in df.iterrows():
        try:
            s = CifParser.from_str(row["cif"]).get_structures(primitive=True)[0]
            structures.append(s)
        except Exception:
            continue

    print(f"Загружено {len(structures)} структур для split={split}")
    return structures
