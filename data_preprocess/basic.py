import lmdb
import os
import pickle
import pickle
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import pandas as pd


def get_canonical_smile(testsmi, isomericSmiles=True):
    try:
        mol = Chem.MolFromSmiles(testsmi)
        canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    except:
        canonical_smi = testsmi
    return canonical_smi


def get_target_order(smiles_target, check=False, add_h=True):
    mol = Chem.MolFromSmiles(smiles_target)
    if add_h:
        mol = AllChem.AddHs(mol)
    atoms = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    assert (not check) or (0 not in atoms)
    return atoms


# ??
def get_atoms(smi, add_h=True):
    mol = Chem.MolFromSmiles(smi)
    if add_h:
        mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
    if not add_h:
        atoms = [i for i in atoms if i != "H"]
    return atoms


def rename(smiles, use_split=False):
    if use_split:
        smiles = smiles.split(".")
        list_ = []
        for i in smiles:
            list_.append(rename(i))
        return ".".join(list_)
    else:
        mol = Chem.MolFromSmiles(smiles)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)


def smi2coord(smiles, path=None):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    if path is not None:
        Chem.MolToMolFile(mol, path)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    return coordinates


def draw_mol(smis, save_path, mols_per_row=4, img_size=(400, 400)):
    mols = []
    for smi in smis:
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            mol = None
        mols.append(mol)
    img = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=img_size, legends=["" for x in mols]
    )
    img.save(save_path)


def check_smiles_atoms_map_fit(smiles_target, target_atoms, target_map=None):
    flag = True
    if target_map is not None:
        assert len(target_map) == len(target_atoms)
        target_order = get_target_order(smiles_target)
        if not (target_order == target_map):
            flag = False
            print(smiles_target)
            print(target_order)
            print(target_map)

    atoms = get_atoms(smiles_target, add_h=True)
    if not (atoms == target_atoms):
        flag = False
        print(smiles_target)
        print(atoms)
        print(target_atoms)

    return flag


def check_smiles_atoms_map_fit_list(smiles_reactant, reactant_atoms, reactant_map):
    list_ = [
        check_smiles_atoms_map_fit(
            smiles_reactant[i], reactant_atoms[i], reactant_map[i]
        )
        for i in range(len(smiles_reactant))
    ]
    return len(list_) - sum(list_)


def csv_file_read(path, usecols=None):
    head_row = pd.read_csv(path, nrows=0)
    print(list(head_row))
    head_row_list = list(head_row)
    if usecols is None:
        usecols=head_row_list
    csv_result = pd.read_csv(path, usecols=usecols)
    row_list = csv_result.values.tolist()
    return row_list


# def check_equal_w_wo_map(smiles):
#     flag = True
#     atoms = get_atoms(smiles, add_h=True)
#     atoms = [i for i in atoms if i !='H']
#     map = get_target_order(smiles)
#     canonical_smile = rename(smiles)
#     canonical_atoms = get_atoms(canonical_smile, add_h=True)
#     canonical_atoms = [i for i in canonical_atoms if i !='H']
#     canonical_map = get_target_order(canonical_smile)

#     print(smiles)
#     print(canonical_smile)
#     print(atoms)
#     print(canonical_atoms)
#     print(map)
#     print(canonical_map)


def get_dictionary(f):
    with open(f, "r", encoding="utf-8") as fd:
        lines = [i.strip() for i in fd.readlines()]
    return set(lines)


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        assert hasattr(self, "env")

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data


def brief_dataset(path, print_all=False):
    dataset = LMDBDataset(path)
    print(len(dataset))
    result = dataset[0]
    print("keys:", result.keys())
    for i in tqdm(range(len(dataset))):
        result = dataset[i]
        print(result)
        break
    return dataset


def make_one_lmdb(path, outputfilename):
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
    if not isinstance(path, list):
        path = [path]
    for p in path:
        d = LMDBDataset(p)
        for i in tqdm(range(len(d))):
            tmp = d[i]
            target = tmp["id"]
            inner_output = pickle.dumps(tmp, protocol=-1)
            txn_write.put(f"{ii}".encode("ascii"), inner_output)
            ii += 1
    txn_write.commit()
    env_new.close()


def rm_h_coordinates(atoms, coordinates):
    atoms2 = [i != "H" for i in atoms]
    atoms3 = [i for i in atoms if i != "H"]
    list_ = []
    for co in coordinates:
        assert co.shape[0] == len(atoms2)
        c = co[atoms2]
        assert c.shape[0] == len(atoms3)
        list_.append(c)
    return atoms3, list_

def rm_h_coordinates_map(target_atoms, target_coordinates, target_map):
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