import numpy as np
from lmdb_preprocess import LMDBDataset
import os
import lmdb
from tqdm import tqdm
import pickle
from basic import get_target_order
from basic import LMDBDataset


def process(result):
    raw_string = result["raw_string"].split(">>")
    smiles_reactant = raw_string[0].split(".")
    smiles_target = raw_string[1]
    result["target_coordinates"] = np.array(result["target_coordinates"])
    result["reactant_coordinates"] = [
        np.array(i) for i in result["reactant_coordinates"]
    ]
    result["target_map"] = get_target_order(smiles_target)
    assert len(result["target_map"]) == result["target_coordinates"].shape[1]
    result["reactant_map"] = [get_target_order(i) for i in smiles_reactant]
    for i in range(len(result["reactant_map"])):
        assert len(result["reactant_map"][i]) == result["reactant_coordinates"][i].shape[1]     
    return result


def make_lmdb_atoms(path, outputfilename):
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
    path = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_2/test.lmdb"
    outputfilename = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_3/test.lmdb"

    make_lmdb_atoms(path, outputfilename)
