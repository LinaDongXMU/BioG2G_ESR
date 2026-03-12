# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import pickle
import logging
from tqdm import tqdm
from basic import LMDBDataset, csv_file_read, get_target_order, rm_h_coordinates_map, get_atoms
import numpy as np
logger = logging.getLogger(__name__)


def get_dict_dataset(path_3d, name):
    dict_3d = {}
    dataset_3d = []
    if not isinstance(path_3d, list):
        path_3d = [path_3d]

    for p in range(len(path_3d)):
        dataset_3d.append(LMDBDataset(path_3d[p]))
        for i in tqdm(range(len(dataset_3d[p]))):
            target = dataset_3d[p][i][name]
            if target not in dict_3d.keys():
                dict_3d[target] = (p, i)
    return dict_3d, dataset_3d


def get_lmdb_writer(outputfilename):
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
    return env_new, txn_write


def make_lmdb(path_3d, path_smi, path_2d, outputfilename):
    try:
        os.remove(outputfilename)
    except:
        pass
    if "lmdb" in path_smi:
        dataset_smi = LMDBDataset(path_smi)
    elif ".csv" in path_smi:
        dataset_smi = csv_file_read(path_smi)
    else:
        raise

    dict_3d, dataset_3d = get_dict_dataset(path_3d, name="id")
    dict_2d, dataset_2d = get_dict_dataset(path_2d, name="smi")
    env_new, txn_write = get_lmdb_writer(outputfilename)

    ii = 0
    coun = 0
    for i in tqdm(range(len(dataset_smi))):
        result = dataset_smi[i]
        if "lmdb" in path_smi:
            raise
            # target = result["raw_string"].split(">>")[1]
        elif ".csv" in path_smi:
            raw_string = result[2]
            target = raw_string.split(">")[-1]
            result = {}
            result["rxn_smiles"] = raw_string
            result["target_map"] = get_target_order(target, check=False, add_h=True)
            result["target_atoms"] = get_atoms(target, add_h=True)
            
        if target in dict_3d.keys():
            p, k = dict_3d[target]
            tmp_result = dataset_3d[p][k]
            assert result["target_atoms"] == tmp_result["atoms"]
            result["target_coordinates"] = tmp_result["coordinates"].copy()
        if target in dict_2d.keys():
            p, k = dict_2d[target]
            tmp_result = dataset_2d[p][k]
            if len(tmp_result["coordinates"]) == 0:
                print("no 2d", target)
            else:
                if target not in dict_3d.keys() or len(result["target_coordinates"]) == 0:
                    assert tmp_result["atoms"] == result["target_atoms"]
                    # result["target_atoms"] = tmp_result["atoms"].copy()
                    result["target_coordinates"] = tmp_result["coordinates"].copy()
                else:
                    assert tmp_result["atoms"] == result["target_atoms"]
                    assert (
                        result["target_coordinates"][0].shape
                        == tmp_result["coordinates"][0].shape
                    )
                    result["target_coordinates"] += tmp_result["coordinates"].copy()
        result["target_coordinates"] = np.array(result["target_coordinates"])
        if result["target_coordinates"].shape[0] > 0:
            target_atoms_tmp, target_coordinates_tmp, target_map_tmp = rm_h_coordinates_map(
                result["target_atoms"], result["target_coordinates"], result["target_map"]
            )
            result["target_atoms"] = target_atoms_tmp
            result["target_coordinates"] = target_coordinates_tmp
            result["target_map"] = target_map_tmp
        if len(result["target_coordinates"]) > 0:
            inner_output = pickle.dumps(result, protocol=-1)
            txn_write.put(f"{ii}".encode("ascii"), inner_output)
            ii += 1
        if (
            len(result["target_coordinates"]) != 11
            and len(result["target_coordinates"]) != 1
        ):
            print(target, len(result["target_coordinates"]))
            coun += 1
        if target not in dict_3d.keys() and target not in dict_2d.keys():
            print("do not have 2d or 3d", target)
    txn_write.commit()
    env_new.close()
    print(ii)
    print(coun)


if __name__ == "__main__":
    pass
    make_lmdb(
        path_3d=["/data/projects/BioG2G/dataset/synthesis/uspto_full_lmdb_cal/uspto_valT.lmdb"],
        path_smi="/data/projects/BioG2G/dataset/synthesis/uspto_full_20230317_200/valid.csv",
        path_2d=["/data/projects/BioG2G/dataset/synthesis/uspto_full_lmdb_cal/uspto_valT_2d.lmdb"],
        outputfilename="/data/projects/BioG2G/dataset/synthesis/uspto_full_20230317_200_lmdb/valid.lmdb",
    )
