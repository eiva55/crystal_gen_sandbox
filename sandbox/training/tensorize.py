"""Превращает pymatgen.Structure в тензорный батч для абляции гипотезы 3.

Работает в декартовых координатах (не в дробных) — эквивариантность
EGNN-слоя сформулирована относительно евклидовых поворотов декартовых
векторов, не относительно произвольных преобразований решётки.
"""
import torch
import numpy as np


def build_atom_type_vocab(structures):
    """Строит словарь атомных номеров по датасету — нужен один общий
    словарь на все структуры, чтобы embedding был согласован между
    вызовами."""
    elements = set()
    for s in structures:
        for site in s:
            elements.add(site.specie.Z)
    vocab = {z: i for i, z in enumerate(sorted(elements))}
    return vocab


def tensorize_structure(structure, vocab, noise_std=0.1, cutoff=5.0, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    atom_types = torch.tensor(
        [vocab[site.specie.Z] for site in structure], dtype=torch.long
    )
    x_clean = torch.tensor(structure.cart_coords, dtype=torch.float32)

    noise = torch.tensor(rng.normal(0, noise_std, size=x_clean.shape), dtype=torch.float32)
    x_noisy = x_clean + noise

    # Граф соседства по cutoff-радиусу через сам pymatgen (учитывает
    # периодические граничные условия корректно, в отличие от наивного
    # попарного перебора декартовых расстояний)
    edges_src, edges_dst = [], []
    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, r=cutoff)
        for n in neighbors:
            edges_src.append(n.index)
            edges_dst.append(i)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    return {
        'atom_types': atom_types,
        'x_noisy': x_noisy,
        'x_clean': x_clean,
        'noise': noise,
        'edge_index': edge_index,
    }
