from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import BioG2G.BioG2G_model.data_preprocess.icp.icp as icp
from lmdb_preprocess import LMDBDataset
import os
import lmdb
from tqdm import tqdm
import pickle
from basic import smi2coord


def transform_matrix(B, T):
    N = B.shape[0]
    C = np.ones((N, 4))
    C[:, 0:3] = B
    C = np.dot(T, C.T).T
    return C[:, 0:3]


def test_best_fit(A, B):
    T, R1, t1 = icp.best_fit_transform(B, A)
    C = transform_matrix(B, T)
    return C


def rewrite_sdf(coord, smiles=None, file_path=None, file_path2=None):
    smi2coord(smiles, file_path)
    with open(file_path, "r") as f:
        context = f.readlines()
    num = int(context[3].split()[0])
    for i in range(num):
        co = coord[i]
        line = context[4 + i]
        # line[3:10] = "*******"
        # line[13:20] = "*******"
        # line[23:30] = "*******"
        # a = "*******"
        x = str(round(co[0], 4)).rjust(7)
        y = str(round(co[1], 4)).rjust(7)
        z = str(round(co[2], 4)).rjust(7)

        line = line[:3] + x + line[10:13] + y + line[20:23] + z + line[30:]
        context[4 + i] = line
    with open(file_path2, "w") as f:
        for i in context:
            f.write(i)


def test_icp():
    smilesa = "CC(C)(C)OC(=O)O[C:12](=[O:13])[O:14][C:15]([CH3:16])([CH3:17])[CH3:18]"
    smilesb = "CC(C)(C)OC(=O)O[C](=[O])[O][C]([CH3])([CH3])[CH3]"
    ca = smi2coord(smilesa, "a.sdf")
    cb = smi2coord(smilesb, "b.sdf")
    new_coord = test_best_fit(ca, cb)
    rewrite_sdf(new_coord, "a.sdf", "aa.sdf")


def get_marked_atoms_idx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    atoms = [i for i in atoms if i != 0]
    atoms_dict = {atom.GetAtomMapNum(): i for i, atom, in enumerate(mol.GetAtoms())}
    idx_tmp = [atoms_dict[i] for i in atoms]
    return atoms, idx_tmp


def get_coord_by_atom_idx(smiles_reactant, smiles_target):
    idx, idx_reactant = get_marked_atoms_idx(smiles_reactant)
    mol = Chem.MolFromSmiles(smiles_target)
    mol = AllChem.AddHs(mol)
    atoms = {atom.GetAtomMapNum(): i for i, atom, in enumerate(mol.GetAtoms())}
    idx_target = [atoms[i] for i in idx]
    return idx_target, idx_reactant


def center_atoms(src_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    return src_atoms - src_center


def align_target(smiles, coordinates):
    _, used_atoms = get_marked_atoms_idx(smiles)
    list_ = []
    first = coordinates[0]
    first = center_atoms(first)
    list_.append(first)

    first_used = first[used_atoms]

    for i in range(1, len(coordinates)):
        atoms_tmp = coordinates[i]
        atoms_tmp_used = atoms_tmp[used_atoms]
        T, R1, t1 = icp.best_fit_transform(atoms_tmp_used, first_used)
        atoms_tmp_new = transform_matrix(atoms_tmp, T)
        # print(np.allclose(atoms_tmp_new, first, atol=0.1))
        list_.append(atoms_tmp_new)
    return list_


def align_reactant_tmp(
    smiles_reactant, smiles_target, coordinates_reactant, coordinates_target
):
    idx_target, idx_reactant = get_coord_by_atom_idx(smiles_reactant, smiles_target)
    list_ = []
    first = coordinates_target[0]
    first_used = first[idx_target]
    for i in range(len(coordinates_reactant)):
        atoms_tmp = coordinates_reactant[i]
        atoms_tmp_used = atoms_tmp[idx_reactant]
        T, R1, t1 = icp.best_fit_transform(atoms_tmp_used, first_used)
        atoms_tmp_new = transform_matrix(atoms_tmp, T)
        list_.append(atoms_tmp_new)
    return list_


def align_reactant(
    smiles_reactant, smiles_target, coordinates_reactant, coordinates_target
):
    smiles_reactant = smiles_reactant.split(".")
    list_ = []
    for index, smiles_react in enumerate(smiles_reactant):
        tmp = align_reactant_tmp(
            smiles_react, smiles_target, coordinates_reactant[index], coordinates_target
        )
        list_.append(tmp)
    return list_


def make_lmdb_align(path, outputfilename):
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
        raw_string = result["raw_string"].split(">>")
        smiles_reactant = raw_string[0]
        smiles_target = raw_string[1]
        result["target_coordinates"] = np.array(
            align_target(smiles_target, result["target_coordinates"])
        )
        result["reactant_coordinates"] = align_reactant(
            smiles_reactant,
            smiles_target,
            result["reactant_coordinates"],
            result["target_coordinates"],
        )
        result["reactant_coordinates"] = [
            np.array(i) for i in result["reactant_coordinates"]
        ]

        # print(result["target_coordinates"][0][:4])
        # print(result["target_coordinates"][1][:4])
        # for idx, i in enumerate(result["target_coordinates"]):
        #     name = "a{}.sdf".format(idx)
        #     rewrite_sdf(i, smiles=smiles_target, file_path=name, file_path2=name)

        # for idx, i in enumerate(result["reactant_coordinates"][0]):
        #     name = "b{}.sdf".format(idx)
        #     rewrite_sdf(i, smiles=smiles_reactant.split(".")[0], file_path=name, file_path2=name)

        # for idx, i in enumerate(result["reactant_coordinates"][1]):
        #     name = "c{}.sdf".format(idx)
        #     rewrite_sdf(i, smiles=smiles_reactant.split(".")[1], file_path=name, file_path2=name)
        # raise
        inner_output = pickle.dumps(result, protocol=-1)
        txn_write.put(f"{ii}".encode("ascii"), inner_output)
        ii += 1
    txn_write.commit()
    env_new.close()
    print(ii)


if __name__ == "__main__":
    path = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_3/test.lmdb"
    outputfilename = "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221204_4/test.lmdb"

    make_lmdb_align(path, outputfilename)
