from rdkit import Chem
from basic import LMDBDataset
from tqdm import tqdm
import lmdb
import pickle
import os
import numpy as np
from basic import get_canonical_smile, draw_mol
from .from_utils.mol_graph_basic import get_adjacency_matrix, test, graph2mol
import copy

np.set_printoptions(threshold=np.inf)
sum_dict = {}
count = [1]


def draw_new(result, name, smiles=[]):
    raw_string = result["raw_string"].split(">>")
    smiles_reactant = raw_string[0]
    smiles_target = raw_string[1]
    print(raw_string)
    draw_mol(
        [smiles_target, smiles_reactant] + smiles,
        "img3/{}_{}.png".format(count[0], name),
        mols_per_row=3,
    )
    count[0] += 1


def collect(dict_, k, v):
    if k in dict_.keys():
        dict_[k] += v
    else:
        dict_[k] = v


def target_process(result, add_h):
    smiles_target = result["raw_string"].split(">>")[1]
    dict_ = get_adjacency_matrix(smiles_target, add_h=add_h)

    assert result["target_atoms"] == dict_["atoms"]
    assert result["target_map"] == dict_["atoms_map"]
    result["target_atoms_chiraltag"] = dict_["atoms_chiraltag"]
    result["target_atoms_charge"] = dict_["atoms_charge"]
    result["target_atom_h_number"] = dict_["atom_h_number"]
    result["target_adjacency_matrix"] = dict_["adjacency_matrix"]
    # result["target_bond_stereo"] = dict_["bond_stereo"]
    # result["target_bond_stereo_dict"] = dict_["bond_stereo_dict"]
    return result


def reactant_process(result, add_h):
    smiles_reactant = result["raw_string"].split(">>")[0].split(".")
    reactant_atoms_chiraltag = []
    reactant_atoms_charge = []
    reactant_atom_h_number = []
    reactant_adjacency_matrix = []
    for i in range(len(smiles_reactant)):
        smiles = smiles_reactant[i]
        dict_ = get_adjacency_matrix(smiles, add_h=add_h)
        assert result["reactant_atoms"][i] == dict_["atoms"]
        assert result["reactant_map"][i] == dict_["atoms_map"]
        reactant_atoms_chiraltag.append(dict_["atoms_chiraltag"])
        reactant_atoms_charge.append(dict_["atoms_charge"])
        reactant_atom_h_number.append(dict_["atom_h_number"])
        reactant_adjacency_matrix.append(dict_["adjacency_matrix"])

    result["reactant_atoms_chiraltag"] = reactant_atoms_chiraltag
    result["reactant_atoms_charge"] = reactant_atoms_charge
    result["reactant_atom_h_number"] = reactant_atom_h_number
    result["reactant_adjacency_matrix"] = reactant_adjacency_matrix
    return result


def new_target_process(result, add_h):
    assert add_h == False or add_h == None
    target_map = result["target_map"]
    assert 0 not in target_map
    target_atoms = result["target_atoms"]
    dict_map = {m: idx for idx, m in enumerate(target_map)}
    new_target_adjacency_matrix = np.zeros_like(result["target_adjacency_matrix"])
    new_target_active_site_matrix = [
        0 for _ in range(len(result["target_atoms_chiraltag"]))
    ]
    new_target_atoms_chiraltag = [
        None for _ in range(len(result["target_atoms_chiraltag"]))
    ]

    for i in range(len(result["reactant_atoms"])):
        a = result["reactant_atoms"][i]
        m = result["reactant_map"][i]
        c = result["reactant_atoms_chiraltag"][i]
        am = result["reactant_adjacency_matrix"][i]
        for j in range(len(a)):
            if m[j] == 0:
                continue
            assert target_atoms[dict_map[m[j]]] == a[j]
            new_target_atoms_chiraltag[dict_map[m[j]]] = c[j]
            for k in range(j + 1, len(a)):
                if m[k] == 0:
                    continue
                assert target_atoms[dict_map[m[k]]] == a[k]
                new_target_adjacency_matrix[dict_map[m[j]], dict_map[m[k]]] = am[j, k]
                new_target_adjacency_matrix[dict_map[m[k]], dict_map[m[j]]] = am[k, j]

        for j in range(len(a)):
            for k in range(len(a)):
                if m[j] != 0 and m[k] == 0 and am[j, k] != 0:
                    new_target_active_site_matrix[dict_map[m[j]]] = 1
                elif m[j] == 0 and m[k] != 0 and am[j, k] != 0:
                    new_target_active_site_matrix[dict_map[m[k]]] = 1

    new_target_active_site_matrix_tmp = abs(
        new_target_adjacency_matrix - result["target_adjacency_matrix"]
    ).sum(1)
    new_target_active_site_matrix = [
        1
        if new_target_active_site_matrix_tmp[i] >= 1
        or new_target_active_site_matrix[i] >= 1
        else 0
        for i in range(len(new_target_active_site_matrix))
    ]
    # if not new_target_atoms_chiraltag == result["target_atoms_chiraltag"]:
    #     # draw_mol([smiles_target] + smiles_reactant, "a.jpg")
    #     print(result["raw_string"])
    #     print(result["target_atoms"])
    #     print(result["target_map"])

    #     print(new_target_atoms_chiraltag)
    #     print(result["target_atoms_chiraltag"])
    #     print(new_target_active_site_matrix)
    #     print("*" * 10)
    #     raise

    result["new_target_atoms_chiraltag"] = new_target_atoms_chiraltag
    result["new_target_active_site_matrix"] = new_target_active_site_matrix
    result["new_target_adjacency_matrix"] = new_target_adjacency_matrix
    return result


def remove(result):
    list_ = [
        "new_target_atoms_chiraltag",
        "new_target_active_site_matrix",
        "new_target_active_site_matrix",
    ]
    for i in list_:
        if i in result.keys():
            result.pop(i)
    return result


def get_new_map(reactant_atoms, reactant_map, add_len=50, mini_idx=1001):
    reactant_map = copy.deepcopy(reactant_map)
    new_atoms = []
    new_map = []
    for i in range(len(reactant_atoms)):
        assert len(reactant_atoms[i]) == len(reactant_map[i])
        for j in range(len(reactant_atoms[i])):
            if reactant_map[i][j] == 0:
                reactant_map[i][j] = mini_idx
                new_atoms.append(reactant_atoms[i][j])
                new_map.append(reactant_map[i][j])
                mini_idx += 1
    pad_len = add_len - len(new_atoms)
    assert pad_len >= 0
    new_atoms = new_atoms + ["[MASK]" for _ in range(pad_len)]
    new_map = new_map + [-1 for _ in range(pad_len)]
    assert len(new_atoms) == add_len
    assert len(new_map) == add_len
    return new_atoms, new_map, reactant_map


def new_target_process_2(result, add_h):
    add_len = 50
    mini_idx = 1001
    assert add_h == False or add_h == None
    target_map = result["target_map"]
    assert 0 not in target_map

    new_target_adjacency_matrix = np.zeros(
        (
            result["target_adjacency_matrix"].shape[0] + add_len,
            result["target_adjacency_matrix"].shape[0] + add_len,
        )
    )
    new_target_atoms_chiraltag = [
        0 for _ in range(len(result["target_atoms_chiraltag"]) + add_len)
    ]

    new_target_atoms_charge = [
        0 for _ in range(len(result["target_atoms_charge"]) + add_len)
    ]

    new_target_atom_h_number = [
        0 for _ in range(len(result["target_atom_h_number"]) + add_len)
    ]

    new_atoms, new_map, reactant_map = get_new_map(
        result["reactant_atoms"], result["reactant_map"], add_len=50, mini_idx=mini_idx
    )

    new_atoms = new_atoms + result["target_atoms"]
    new_map = new_map + result["target_map"]
    dict_map = {m: idx for idx, m in enumerate(new_map)}

    for i in range(len(result["reactant_atoms"])):
        a = result["reactant_atoms"][i]
        c = result["reactant_atoms_chiraltag"][i]
        ac = result["reactant_atoms_charge"][i]
        am = result["reactant_adjacency_matrix"][i]
        ah = result["reactant_atom_h_number"][i]
        m = reactant_map[i]

        for j in range(len(a)):
            assert new_atoms[dict_map[m[j]]] == a[j]
            new_target_atoms_chiraltag[dict_map[m[j]]] = c[j]
            new_target_atoms_charge[dict_map[m[j]]] = ac[j]
            new_target_atom_h_number[dict_map[m[j]]] = ah[j]
            for k in range(j + 1, len(a)):
                assert new_atoms[dict_map[m[k]]] == a[k]
                new_target_adjacency_matrix[dict_map[m[j]], dict_map[m[k]]] = am[j, k]
                new_target_adjacency_matrix[dict_map[m[k]], dict_map[m[j]]] = am[k, j]

    result["new2_target_atoms_chiraltag"] = new_target_atoms_chiraltag
    result["new2_target_atoms_charge"] = new_target_atoms_charge
    result["new2_target_atom_h_number"] = new_target_atom_h_number
    result["new2_target_adjacency_matrix"] = new_target_adjacency_matrix
    result["new2_target_atoms"] = new_atoms
    result["new2_target_map"] = new_map
    result["new2_src_atoms_chiraltag"] = [0 for _ in range(add_len)] + result[
        "target_atoms_chiraltag"
    ]
    result["new2_src_atoms_charge"] = [0 for _ in range(add_len)] + result[
        "target_atoms_charge"
    ]
    result["new2_src_atom_h_number"] = [0 for _ in range(add_len)] + result[
        "target_atom_h_number"
    ]
    # print(result["new2_target_atoms"])
    # print(result["new2_target_map"])
    # print(result["new2_target_atoms_chiraltag"])
    # print(result["new2_target_adjacency_matrix"])
    # raise
    return result


def new_target_process_3(result, add_h):
    add_len = 50
    result["new3_target_atoms_chiraltag"] = result["new2_target_atoms_chiraltag"][
        add_len:
    ]
    result["new3_target_atoms_charge"] = result["new2_target_atoms_charge"][add_len:]
    result["new3_target_atom_h_number"] = result["new2_target_atom_h_number"][add_len:]
    result["new3_target_adjacency_matrix"] = np.array(
        result["new2_target_adjacency_matrix"], dtype=np.dtype('int_')
    )[add_len:, add_len:]
    result["new3_target_adjacency_matrix"] = result[
        "new3_target_adjacency_matrix"
    ].tolist()
    new_smiles = graph2mol(
        adjacency_matrix=result["new3_target_adjacency_matrix"],
        atoms=result["target_atoms"],
        atoms_map=result["target_map"],
        atoms_chiraltag=result["new3_target_atoms_chiraltag"],
        atoms_charge=result["new3_target_atoms_charge"],
        bond_stereo=None,
        bond_stereo_dict=None,
        atom_h_number=result["new3_target_atom_h_number"],
    )
    draw_new(result, 0, smiles=[new_smiles])
    new_target_active_site_matrix = np.zeros(len(result["target_atoms_chiraltag"]))
    new_target_active_site_matrix += abs(
        np.array(result["new3_target_atoms_charge"])
        - np.array(result["target_atoms_charge"])
    )
    new_target_active_site_matrix += abs(
        np.array(result["new3_target_atom_h_number"])
        - np.array(result["target_atom_h_number"])
    )
    new_target_active_site_matrix += abs(
        result["new3_target_adjacency_matrix"] - result["target_adjacency_matrix"]
    ).sum(1)
    new_target_active_site_matrix[new_target_active_site_matrix >= 1] = 1
    result["new_target_active_site_matrix"] = new_target_active_site_matrix
    sum_ = sum(new_target_active_site_matrix)
    # if sum_>10:
    #     print(count)
    #     print(new_target_active_site_matrix)
    #     draw_new(result, sum_)
    collect(sum_dict, sum_, 1)

    return result


def process(result, add_h):
    result = target_process(result, add_h=add_h)
    result = reactant_process(result, add_h=add_h)
    result = remove(result)
    result = new_target_process_2(result, add_h=add_h)
    result = new_target_process_3(result, add_h=add_h)
    return result


def make_lmdb_graph(path, outputfilename, add_h):
    dataset_smi = LMDBDataset(path)
    try:
        os.remove(outputfilename)
    except:
        pass

    env_new = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    ii = 0
    for i in tqdm(range(len(dataset_smi))):
        result = dataset_smi[i]
        result = process(result, add_h=add_h)
        inner_output = pickle.dumps(result, protocol=-1)
        txn_write.put(f"{ii}".encode("ascii"), inner_output)
        ii += 1
    txn_write.commit()
    env_new.close()
    print(ii)


def test_lmdb_graph(path, add_h=False):
    dataset_smi = LMDBDataset(path)
    print(len(dataset_smi))
    flag_target = 0
    flag_reactant = 0
    for i in tqdm(range(len(dataset_smi))):
        result = dataset_smi[i]
        raw_string = result["raw_string"].split(">>")
        smiles_reactant = raw_string[0]
        # smiles_reactant = smiles_reactant.split(".")
        smiles_target = raw_string[1]
        # tmp_a = get_charge_dict(smiles_reactant)
        # tmp_b = get_charge_dict(smiles_target)
        # flag = tmp_a == tmp_b
        flag = test(smiles_reactant)
        if not flag:
            flag_reactant += 1
        flag = test(smiles_target)
        if not flag:
            flag_target += 1
            # draw_mol([smiles_target, smiles_reactant], "img/{}.png".format(flag_all), mols_per_row=2)
            # print(smiles_target)
    print(flag_reactant, flag_target)


if __name__ == "__main__":
    # test()
    path = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221207_1_rmh/test.lmdb"
    outputfilename = (
        "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20230102_1_rmh/test.lmdb"
    )
    make_lmdb_graph(path, outputfilename, add_h=None)
    # test_lmdb_graph(outputfilename)
    print(sum_dict)
