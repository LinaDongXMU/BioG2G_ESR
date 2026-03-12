from rename_match_smiles import rename, sort_mol
from basic import LMDBDataset
from lmdb_preprocess import *


def make_lmdb_reactant(path_3d, path_2d, path_smi, outputfilename):

    dataset_smi = LMDBDataset(path_smi)

    dict_3d = {}
    dataset_3d = []
    if not isinstance(path_3d, list):
        path_3d = [path_3d]

    for p in range(len(path_3d)):
        dataset_3d.append(LMDBDataset(path_3d[p]))
        for i in tqdm(range(len(dataset_3d[p]))):
            target = dataset_3d[p][i]["id"]
            if target not in dict_3d.keys():
                dict_3d[target] = (p, i)

    dict_2d = {}
    dataset_2d = []
    if not isinstance(path_2d, list):
        path_2d = [path_2d]

    for p in range(len(path_2d)):
        dataset_2d.append(LMDBDataset(path_2d[p]))
        for i in tqdm(range(len(dataset_2d[p]))):
            target = dataset_2d[p][i]["smi"]
            if target not in dict_2d.keys():
                dict_2d[target] = (p, i)

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
        reactant = result["raw_string"].split(">>")[0].split(".")
        result["reactant_atoms"] = []
        result["reactant_coordinates"] = []
        for target in reactant:
            if target in dict_3d.keys():
                p, k = dict_3d[target]
                tmp_result = dataset_3d[p][k]
                # if len(tmp_result["coordinates"]) != 10:
                #     print(tmp_result["id"], len(tmp_result["coordinates"]))
                tmp_target_atoms = tmp_result["atoms"].copy()
                tmp_target_coordinates = tmp_result["coordinates"].copy()

            if target in dict_2d.keys():
                p, k = dict_2d[target]
                tmp_result = dataset_2d[p][k]
                if target not in dict_3d.keys():
                    tmp_target_atoms = tmp_result["atoms"].copy()
                    tmp_target_coordinates = tmp_result["coordinates"].copy()
                else:
                    assert tmp_result["atoms"] == tmp_target_atoms
                    assert (
                        tmp_target_coordinates[0].shape
                        == tmp_result["coordinates"][0].shape
                    )
                    tmp_target_coordinates += tmp_result["coordinates"].copy()

            if len(tmp_target_coordinates) != 11 and len(tmp_target_coordinates) != 1:
                print(result["target_id"], len(tmp_target_coordinates))

            if target not in dict_3d.keys() and target not in dict_2d.keys():
                print(target)
            result["reactant_atoms"].append(tmp_target_atoms)
            result["reactant_coordinates"].append(tmp_target_coordinates)

        inner_output = pickle.dumps(result, protocol=-1)
        txn_write.put(f"{ii}".encode("ascii"), inner_output)
        ii += 1
    txn_write.commit()
    env_new.close()
    print(ii)


if __name__ == "__main__":
    make_lmdb_reactant(
        path_3d=["/data/users/yaolin/BioG2G_/retro/reactant/USPTO50K_3D_raw2.lmdb"],
        path_smi="/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_1/train.lmdb",
        path_2d=["/data/users/yaolin/BioG2G_/retro/reactant/USPTO50K_2D_raw.lmdb"],
        outputfilename="/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_2/train.lmdb",
    )
