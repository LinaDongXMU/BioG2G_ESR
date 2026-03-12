import numpy as np
import os
import lmdb
from tqdm import tqdm
import pickle
from basic import get_target_order
from basic import LMDBDataset


def get_result(target_atoms, target_coordinates, target_map):
    assert (
        len(target_atoms) == len(target_map)
        and len(target_atoms) == target_coordinates.shape[1]
    )
    target_atoms_tmp = [i for i in target_atoms if i != "H"]
    idx = [i != "H" for i in target_atoms]
    target_coordinates_tmp = target_coordinates[:, idx]
    target_map_tmp = [
        target_map[i] for i in range(len(target_atoms)) if target_atoms[i] != "H"
    ]
    assert len(target_atoms_tmp) == len(target_map_tmp) and len(target_atoms_tmp) == (
        target_coordinates_tmp.shape[1]
    )
    return target_atoms_tmp, target_coordinates_tmp, target_map_tmp


def get_result_reactant(target_atoms, target_coordinates, target_map):
    a_list = []
    c_list = []
    m_list = []
    assert len(target_atoms) == len(target_coordinates) and len(target_map) == len(target_atoms)
    for i in range(len(target_atoms)):
        a = target_atoms[i]
        c = target_coordinates[i]
        m = target_map[i]
        a, c, m = get_result(a, c, m)
        a_list.append(a)
        c_list.append(c)
        m_list.append(m)
    return a_list, c_list, m_list


def process(result):
    target_atoms = result["target_atoms"]
    target_coordinates = result["target_coordinates"]
    target_map = result["target_map"]

    (
        result["target_atoms"],
        result["target_coordinates"],
        result["target_map"],
    ) = get_result(target_atoms, target_coordinates, target_map)

    reactant_atoms = result["reactant_atoms"]
    reactant_coordinates = result["reactant_coordinates"]
    reactant_map = result["reactant_map"]
    (
        result["reactant_atoms"],
        result["reactant_coordinates"],
        result["reactant_map"],
    ) = get_result_reactant(reactant_atoms, reactant_coordinates, reactant_map)

    return result


def make_lmdb_remove_h(path, outputfilename):
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
        result = process(result)
        inner_output = pickle.dumps(result, protocol=-1)
        txn_write.put(f"{ii}".encode("ascii"), inner_output)
        ii += 1
    txn_write.commit()
    env_new.close()
    print(ii)


if __name__ == "__main__":
    path = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_4/train.lmdb"
    outputfilename = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_5/train.lmdb"

    make_lmdb_remove_h(path, outputfilename)
