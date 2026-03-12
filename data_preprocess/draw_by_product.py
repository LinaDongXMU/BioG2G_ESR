import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
from queue import Queue


def by_product_index(mol, start_idx):
    q = Queue()
    q.put(start_idx)
    using_atom = set([start_idx])
    stop_atom = set()
    while not q.empty():
        idx = q.get()
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in using_atom:
                continue
            if neighbor.GetAtomMapNum() > 0:
                stop_atom.add(neighbor_idx)
            else:
                q.put(neighbor_idx)
                using_atom.add(neighbor_idx)
    for idx in range(len(mol.GetAtoms()) - 1, -1, -1):
        if idx not in using_atom:
            if idx not in stop_atom:
                mol.RemoveAtom(idx)
            else:
                mol.ReplaceAtom(idx, Chem.Atom(0))

    return using_atom, mol


def get_by_product(smi):
    precursor = Chem.MolFromSmiles(smi)
    using_atoms = set()
    parts = []
    for i in range(len(precursor.GetAtoms())):
        if i not in using_atoms and precursor.GetAtomWithIdx(i).GetAtomMapNum() == 0:
            using_atom, mol = by_product_index(Chem.rdchem.RWMol(precursor), i)
            using_atoms = using_atoms | using_atom
            parts.append(Chem.MolToSmiles(mol))
    return parts


def draw_mol(smis, save_path):
    mols = []
    for smi in smis:
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            return
        mols.append(mol)
    img = Draw.MolsToGridImage(
        mols, molsPerRow=4, subImgSize=(400, 400), legends=["" for x in mols]
    )
    img.save(save_path)


def csv_file_read(path):
    head_row = pd.read_csv(path, nrows=0)
    print(list(head_row))
    head_row_list = list(head_row)

    csv_result = pd.read_csv(path, usecols=head_row_list)
    row_list = csv_result.values.tolist()
    return row_list


def test_():
    s = "CC(C)(C)OC(=O)O[C:12](=[O:13])[O:14][C:15]([CH3:16])([CH3:17])[CH3:18].[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][c:7]2[c:8]([cH:9][cH:10][nH:11]2)[cH:19]1"
    a = get_by_product(s)
    draw_mol(a, "a.jpg")


def print_part(part):
    tmp = [len(Chem.MolFromSmiles(i).GetAtoms()) for i in part]
    return sum(tmp)


def get_img(path1):
    row_list = csv_file_read(path1)
    max_len = 0
    part_atoms = 0

    for i in tqdm(row_list):
        result = i[2]
        id = i[1]
        result = result.split(">>")
        reactant = result[0].split(".")
        target = result[1]
        part = []
        for k in reactant:
            part += get_by_product(k)
        num_part = print_part(part)
        part_atoms = num_part if num_part > part_atoms else part_atoms
        len_r = len(reactant)
        # if len_r > 2:
        #     print(id)
        max_len = len_r if len_r > max_len else max_len
        result = [target] + reactant + part
        save_path = "img/" + id + ".jpg"
        # draw_mol(result, save_path)
    print(max_len)
    print(part_atoms)


path1 = "/data/users/yaolin/MechRetro/data/USPTO50K/raw/valid.csv"
get_img(path1)
