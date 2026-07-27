import json

import pandas as pd
import torch
import torch.nn.functional as F

import pyxtal
from pyxtal.symmetry import Group


def generate_space_group_embeddings():
    lattice_cell_centerings = ["P", "A", "C", "I", "R", "F"]
    crystal_families = [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "hexagonal",
        "cubic",
    ]
    point_group_symbols = [
        "1",
        "-1",
        "2",
        "m",
        "2/m",
        "222",
        "mm2",
        "mmm",
        "4",
        "-4",
        "4/m",
        "422",
        "4mm",
        "-42m",
        "4/mmm",
        "3",
        "-3",
        "32",
        "3m",
        "-3m",
        "6",
        "-6",
        "6/m",
        "622",
        "6mm",
        "-62m",
        "6/mmm",
        "23",
        "m-3",
        "432",
        "-43m",
        "m-3m",
    ]

    space_group_embeddings_dict = {}
    for space_group_number in range(1, 231):
        hall_number = get_hall_number(space_group_number)
        pyxtal_space_group = Group(hall_number, use_hall=True)

        lattice_cell_centering = pyxtal_space_group.symbol[0]  # P, A, C, I, R, or F
        index = lattice_cell_centerings.index(lattice_cell_centering)
        lattice_cell_centering_emb = F.one_hot(
            torch.LongTensor([index]), num_classes=len(lattice_cell_centerings)
        ).squeeze()

        crystal_family = pyxtal_space_group.lattice_type
        index = crystal_families.index(crystal_family)
        crystal_family_emb = F.one_hot(
            torch.LongTensor([index]), num_classes=len(crystal_families)
        ).squeeze()

        point_group_symbol = pyxtal.symmetry.get_point_group(space_group_number)[0]
        index = point_group_symbols.index(point_group_symbol)
        point_group_emb = F.one_hot(
            torch.LongTensor([index]), num_classes=len(point_group_symbols)
        ).squeeze()

        chiral_emb = torch.LongTensor([int(pyxtal_space_group.chiral)])  # bool
        centrosymmetric_emb = torch.LongTensor([int(pyxtal_space_group.inversion)])  # bool

        space_group_embedding = torch.cat(
            (
                lattice_cell_centering_emb,
                crystal_family_emb,
                point_group_emb,
                chiral_emb,
                centrosymmetric_emb,
            )
        )  # shape (46,)
        space_group_embeddings_dict[str(space_group_number)] = space_group_embedding

    def list_duplicates(list_of_tensors):
        sets_of_space_groups_with_same_embeddings = []
        seen_space_groups = []
        for i in range(0, 229):
            t1 = list_of_tensors[i]
            space_groups_with_same_embedding_as_space_group_i = [i + 1]
            for j in range(i + 1, 230):
                t2 = list_of_tensors[j]
                if j + 1 not in seen_space_groups and torch.equal(t1, t2):
                    space_groups_with_same_embedding_as_space_group_i.append(j + 1)
            space_groups_with_same_embedding_as_space_group_i = set(
                space_groups_with_same_embedding_as_space_group_i
            )
            seen_space_groups.extend(list(space_groups_with_same_embedding_as_space_group_i))

            if len(space_groups_with_same_embedding_as_space_group_i) > 1:
                if (
                    set(space_groups_with_same_embedding_as_space_group_i)
                    not in sets_of_space_groups_with_same_embeddings
                ):
                    sets_of_space_groups_with_same_embeddings.append(
                        set(space_groups_with_same_embedding_as_space_group_i)
                    )
        return sets_of_space_groups_with_same_embeddings

    # Check for collisions between space group embeddings
    sets_of_space_groups_with_same_embeddings = list_duplicates(
        list(space_group_embeddings_dict.values())
    )
    print(
        "Before adjustment, number of collision points in space group embedding space: {}".format(
            len(sets_of_space_groups_with_same_embeddings)
        )
    )  # 55 collision points
    print(sets_of_space_groups_with_same_embeddings)
    print(
        "Embedding size before adjustment: {}".format(space_group_embeddings_dict["1"].shape)
    )

    def max_set_length(list_of_sets):
        max_set_length = 0
        for set in list_of_sets:
            if len(set) > max_set_length:
                max_set_length = len(set)
        return max_set_length

    max_space_groups_per_collision_point = max_set_length(
        sets_of_space_groups_with_same_embeddings
    )
    # At most 16 duplicates at a given collision point

    # Concatenate length-16 one-hot encodings to each space group embedding
    # to ensure each space group embedding is unique. Otherwise, the embeddings
    # will be ignorant to screw/glide symmetries.
    for space_group_number in range(1, 231):
        space_group_embedding = torch.LongTensor(
            space_group_embeddings_dict[str(space_group_number)]
        )

        for collision_point_set in sets_of_space_groups_with_same_embeddings:
            if space_group_number in collision_point_set:
                colliding_space_groups = list(collision_point_set)
                index = colliding_space_groups.index(space_group_number)
                extra_embedding_dims = F.one_hot(
                    torch.LongTensor([index]), num_classes=max_space_groups_per_collision_point
                ).squeeze()
                space_group_embedding = torch.cat(
                    (space_group_embedding, extra_embedding_dims)
                )
                space_group_embeddings_dict[str(space_group_number)] = space_group_embedding
                continue

        # Ensure all space group embeddings have the same length
        if space_group_embeddings_dict[str(space_group_number)].shape[0] != 62:
            space_group_embeddings_dict[str(space_group_number)] = torch.cat(
                (
                    space_group_embeddings_dict[str(space_group_number)],
                    torch.zeros(max_space_groups_per_collision_point, dtype=torch.long),
                )
            )

    # Check again for collisions between space group embeddings
    sets_of_space_groups_with_same_embeddings = list_duplicates(
        list(space_group_embeddings_dict.values())
    )
    print(
        "After adjustment, number of collision points in space group embedding space: {}".format(
            len(sets_of_space_groups_with_same_embeddings)
        )
    )  # 0 collision points
    print(sets_of_space_groups_with_same_embeddings)  # empty list
    print("Embedding size after adjustment: {}".format(space_group_embeddings_dict["1"].shape))

    # Convert dictionary values from tensor to JSON serializable lists
    space_group_embeddings_dict = {
        k: v.tolist() for k, v in space_group_embeddings_dict.items()
    }

    with open("space_group_embeddings_62dim.json", "w") as file:
        json.dump(space_group_embeddings_dict, file)


def get_hall_number(space_group_number: int):
    """
    Get Hall number for space group 'space_group_number' (1-230) that
    corresponds to the conventional unit cell origin choice in 'exact_cuts.py'
    """

    # Hall numbers (1-530) from pyxtal.database.HM_Full.csv
    df = pd.read_csv("HM_Full.csv")

    # Space group setting codes following International Tables Vol. B
    # (Shmueli 2001)
    space_group_settings = [
        "1",
        "2",
        "3:b",
        "4:b",
        "5:b1",
        "6:b",
        "7:b1",
        "8:b1",
        "9:b1",
        "10:b",
        "11:b",
        "12:b1",
        "13:b1",
        "14:b1",
        "15:b1",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48:2",
        "49",
        "50:2",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59:2",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68:2",
        "69",
        "70:2",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85:2",
        "86:2",
        "87",
        "88:2",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "122",
        "123",
        "124",
        "125:2",
        "126:2",
        "127",
        "128",
        "129:2",
        "130:2",
        "131",
        "132",
        "133:2",
        "134:2",
        "135",
        "136",
        "137:2",
        "138:2",
        "139",
        "140",
        "141:2",
        "142:2",
        "143",
        "144",
        "145",
        "146:H",
        "147",
        "148:H",
        "149",
        "150",
        "151",
        "152",
        "153",
        "154",
        "155:H",
        "156",
        "157",
        "158",
        "159",
        "160:H",
        "161:H",
        "162",
        "163",
        "164",
        "165",
        "166:H",
        "167:H",
        "168",
        "169",
        "170",
        "171",
        "172",
        "173",
        "174",
        "175",
        "176",
        "177",
        "178",
        "179",
        "180",
        "181",
        "182",
        "183",
        "184",
        "185",
        "186",
        "187",
        "188",
        "189",
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "196",
        "197",
        "198",
        "199",
        "200",
        "201:2",
        "202",
        "203:2",
        "204",
        "205",
        "206",
        "207",
        "208",
        "209",
        "210",
        "211",
        "212",
        "213",
        "214",
        "215",
        "216",
        "217",
        "218",
        "219",
        "220",
        "221",
        "222:2",
        "223",
        "224:2",
        "225",
        "226",
        "227:2",
        "228:2",
        "229",
        "230",
    ]

    pyxtal_hall_number = int(
        df.loc[df["Spg_full"] == space_group_settings[space_group_number - 1]]["Hall"].iloc[0]
    )
    return pyxtal_hall_number


if __name__ == "__main__":
    generate_space_group_embeddings()
