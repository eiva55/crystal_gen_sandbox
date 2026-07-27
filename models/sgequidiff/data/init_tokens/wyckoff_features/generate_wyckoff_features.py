import json

import pyxtal
from pyxtal.symmetry import Group as PyxtalGroup
import torch
import torch.nn.functional as F

import global_vars
from utils.io_utils import get_project_dir
from aliases import *


def embedding_is_unique(
    embedding: List[int],
    embeddings_history: List[List[int]]
) -> bool:
    for i, emb in enumerate(embeddings_history):
        if all([x == y for (x, y) in zip(embedding, emb)]):
            print(f'Collided with embedding {i}')
            return False
    return True


def generate_wyckoff_features():
    # -- Build a hashmap across all Wyckoff position features, and build a dict
    #   of raw Wyckoff position features
    hashmap = {
        'multiplicities': set(),
        'rotations': set(),
        'translations': set(),
        'site symmetry': set(),
    }
    wyckoff_raw_features = {}
    num_wyckoffs = 0
    for sg_num in range(1, 231):
        pyxtal_space_group = PyxtalGroup(sg_num, style='pyxtal')
        for i, wyckoff_position in enumerate(pyxtal_space_group.Wyckoff_positions):
            num_wyckoffs += 1
            wyckoff_letter = wyckoff_position.letter
            multiplicity = wyckoff_position.multiplicity
            rotations, translations = [], []
            for op in wyckoff_position.ops:
                rotations.append(op.rotation_matrix)
                translations.append(op.translation_vector)
                hashmap['rotations'].add(str(op.rotation_matrix))
                hashmap['translations'].add(str(op.translation_vector))

            wyckoff_position.get_site_symmetry()  # fills .site_symm property
            hashmap['multiplicities'].add(multiplicity)
            hashmap['site symmetry'].add(wyckoff_position.site_symm)
            wyckoff_raw_features[sg_num] = {
                wyckoff_letter: {
                    'multiplicity': multiplicity,
                    'rotations': rotations,
                    'translations': translations,
                    'dof': wyckoff_position.get_dof(),
                    'site symmetry': wyckoff_position.site_symm
                }
            }
            if any([glide in wyckoff_position.site_symm for glide in ['a', 'b', 'c', 'n', 'd', 'e']]):
                print('yes')
            if sg_num == 212:
                print(wyckoff_position.site_symm)

    print(num_wyckoffs)
    # 1731
    print(len(hashmap['multiplicities']), max(hashmap['multiplicities']), min(hashmap['multiplicities']))
    # 17, 192, 1
    print(len(hashmap['rotations']))
    # 185
    print(len(hashmap['translations']))
    # 584
    print(len(hashmap['site symmetry']))
    # 100

    # space group + site symmetry symbol + dof + multiplicity
    # 62 + 100 + 4 + 17 = 183
    PROJECT_DIR = get_project_dir()
    space_group_embeddings_path = (
        PROJECT_DIR +
        '/data/init_tokens/space_group_features/space_group_embeddings_62dim.json'
    )
    with open(space_group_embeddings_path, 'r') as f:
        space_group_embeddings: dict = json.load(f)

    freqs = 2 * torch.pi * torch.linspace(
        1.0, 32.0, steps=8, dtype=torch.float,
    )
    wyckoff_embeddings = dict()
    wyckoff_embeddings_dense_array = []
    for sg_num in range(1, 231):
        wyckoff_embeddings[sg_num] = {}
        pyxtal_space_group = PyxtalGroup(sg_num, style='pyxtal')
        for i, wyckoff_position in enumerate(pyxtal_space_group.Wyckoff_positions):
            space_group_embedding = torch.tensor(
                space_group_embeddings[str(sg_num)], dtype=torch.long
            )

            wyckoff_position.get_site_symmetry()  # fills .site_symm property
            site_symmetry_index = list(hashmap["site symmetry"]).index(wyckoff_position.site_symm)
            site_symmetry_embedding = F.one_hot(
                torch.tensor([site_symmetry_index], dtype=torch.long),
                num_classes=len(hashmap["site symmetry"])
            ).squeeze()

            wyckoff_dim = wyckoff_position.get_dof()
            dof_index = [0, 1, 2, 3].index(wyckoff_dim)
            dof_embedding = F.one_hot(
                torch.tensor([dof_index], dtype=torch.long),
                num_classes=4
            ).squeeze()

            multiplicity = list(hashmap["multiplicities"]).index(wyckoff_position.multiplicity)
            multiplicity_embedding = F.one_hot(
                torch.tensor([multiplicity], dtype=torch.long),
                num_classes=len(hashmap["multiplicities"])
            ).squeeze()

            # Algo to construct coordinates embedding:
            # 1. Get midpoint/vertices of each Wyckoff convex shapes in ASU
            # 2. Compute Fourier features from those points
            # 3. Average across points in the Wyckoff position
            wyckoff_letter = wyckoff_position.letter
            print(f'-- {sg_num}, {wyckoff_letter} --')
            if wyckoff_dim == 0:
                wyckoff_points: Tensor = torch.tensor(
                    global_vars.asu_wyckoff_dict[str(sg_num)][wyckoff_letter]["vertices"].astype("float32"),
                    dtype=torch.float
                )
                # (1, 3)
            elif wyckoff_dim in [1, 2]:
                wyckoff_midpoints: Tensor = torch.cat([
                    torch.tensor(
                        shape.astype("float32"), dtype=torch.float
                    ).mean(dim=0, keepdim=True)
                    for shape in global_vars.asu_wyckoff_dict[
                        str(sg_num)][wyckoff_letter]["vertices"]]
                )
                # (n_shapes, 3)
                wyckoff_vertices = torch.cat([
                    torch.tensor(shape.astype("float32"), dtype=torch.float)
                    for shape in global_vars.asu_wyckoff_dict[
                        str(sg_num)][wyckoff_letter]["vertices"]]
                )
                # (n_vertices, 3)
                wyckoff_points = torch.cat(
                    [wyckoff_midpoints, wyckoff_vertices], dim=0
                )
            else:
                wyckoff_midpoints: Tensor = torch.tensor(
                    global_vars.asu_wyckoff_dict[str(sg_num)][wyckoff_letter][
                        "vertices"].astype("float32"), dtype=torch.float
                ).mean(dim=0, keepdim=True)
                # (1, 3)
                wyckoff_vertices = torch.tensor(
                    global_vars.asu_wyckoff_dict[str(sg_num)][wyckoff_letter][
                        "vertices"].astype("float32"), dtype=torch.float
                )
                wyckoff_points = torch.cat(
                    [wyckoff_midpoints, wyckoff_vertices], dim=0
                )
            position_harmonics = wyckoff_points.unsqueeze(-1) * freqs[None, None, :]
            # (n_points, 3, len(freqs))
            coordinates_embedding: List[float] = torch.cat(
                [position_harmonics.sin(), position_harmonics.cos()],
                dim=-1
            ).mean(dim=0).view(-1)
            # (3 * 2 * len(freqs),)

            wyckoff_embedding: List[int] = torch.cat([
                space_group_embedding,      # 62
                site_symmetry_embedding,    # 100
                dof_embedding,              # 4
                multiplicity_embedding,     # 17
                coordinates_embedding       # 3 * number of harmonics = 48
            ]).tolist()
            wyckoff_embeddings[sg_num][wyckoff_position.letter] = wyckoff_embedding
            assert embedding_is_unique(wyckoff_embedding, wyckoff_embeddings_dense_array), f'{sg_num}, {wyckoff_letter}'
            wyckoff_embeddings_dense_array.append(wyckoff_embedding)

    print(len(wyckoff_embedding))  # 231 dimensions. Wren used 444.
    with open(PROJECT_DIR + "/data/init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json", "w") as file:
        json.dump(wyckoff_embeddings, file)


if __name__ == '__main__':
    generate_wyckoff_features()
