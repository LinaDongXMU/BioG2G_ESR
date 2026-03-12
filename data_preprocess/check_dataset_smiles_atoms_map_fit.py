from tqdm import tqdm
from basic import (
    LMDBDataset,
    check_smiles_atoms_map_fit,
    check_smiles_atoms_map_fit_list,
)


def check(path, print_all=False):
    dataset = LMDBDataset(path)
    print(len(dataset))
    result = dataset[0]
    print("keys:", result.keys())

    target_max_atoms = 0
    valid_no_count = 0
    valid_no_count_reactant = 0
    for i in tqdm(range(len(dataset))):
        result = dataset[i]
        target_length = len(result["target_atoms"])
        raw_string = result["raw_string"].split(">>")
        smiles_reactant = raw_string[0]
        smiles_reactant = smiles_reactant.split(".")
        smiles_target = raw_string[1]
        flag = check_smiles_atoms_map_fit(
            smiles_target, result["target_atoms"], result["target_map"]
        )
        if not flag:
            valid_no_count += 1
        flag2 = check_smiles_atoms_map_fit_list(
            smiles_reactant, result["reactant_atoms"], result["reactant_map"]
        )
        valid_no_count_reactant += flag2

        if target_length > target_max_atoms:
            target_max_atoms = target_length
    print(target_max_atoms)
    print(valid_no_count)
    print(valid_no_count_reactant)


if __name__ == "__main__":
    # path = "/data/users/yaolin/BioG2G_/retro/reactant/USPTO50K_3D.lmdb"
    path = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221207_1_rmh/train.lmdb"
    check(path)
