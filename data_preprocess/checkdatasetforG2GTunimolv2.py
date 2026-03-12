import os
import lmdb
from tqdm import tqdm
from basic import LMDBDataset
import pickle
import pandas as pd


def make_lmdb(path, outputfilename, want_list=[]):
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
        result_new = {}
        for k in want_list:
            if k != "raw_string":
                result_new[k] = result[k]
            else:
                result_new["rxn_smiles"] = result[k]

        inner_output = pickle.dumps(result_new, protocol=-1)
        txn_write.put(f"{ii}".encode("ascii"), inner_output)
        ii += 1
    txn_write.commit()
    env_new.close()
    print(ii)


# def read_csv(path):
#     csv_file = pd.read_csv(path)
#     for i in range(len(csv_file)):

#         return dict(self.csv_file.iloc[idx])
path = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20230101_1_rmh/valid.lmdb"
save_path = "/data/users/yaolin/BioG2G_/retro/USPTO50K_brief_20230227/valid.lmdb"
check(path)
# want_list = ["raw_string", "class", "target_atoms", "target_coordinates", "target_map"]
# make_lmdb(path, outputfilename=save_path, want_list=want_list)
