from basic import LMDBDataset
from tqdm import tqdm
import csv
from rdkit import Chem
import pandas as pd
import collections
from rdkit.Chem import AllChem
import sys
sys.path.append("/data/users/yaolin/BioG2G/BioG2G_model")
from BioG2G.utils.graph_process import process_one
from basic import draw_mol

idxx = 0


def get_edge(smiles):
    result = process_one(smiles)
    atoms_map = result["atoms_map"]
    edge_index = result["edge_index"]
    tmp = [
        (atoms_map[edge_index[0][i]], atoms_map[edge_index[1][i]])
        for i in range(len(edge_index[0]))
    ]
    tmp = [i for i in tmp if 0 not in i]
    return set(tmp)


def check_edge(product, reactant, N=5):
    global idxx
    a = get_edge(reactant)
    b = get_edge(product)
    if len((a | b) - (b & a)) >= 2 * N:
        print("aa", idxx, (a | b) - (b & a), len((a | b) - (b & a)))
        # print(smiles)
        # draw_mol([reactant, product], "{}.png".format(idxx))
        idxx += 1
        print(idxx)
        return False
    return True


def get_atoms(testsmi):
    mol = Chem.MolFromSmiles(testsmi)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return atoms


def error(testsmi):
    if testsmi is None or testsmi == "":
        return False
    try:
        mol = Chem.MolFromSmiles(testsmi)
        if mol is None:
            return False
        canonical_smi = Chem.MolToSmiles(mol)
        if canonical_smi is None or canonical_smi == "":
            return False
        return True
    except:
        return False


def get_map(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # mol = AllChem.RemoveHs(mol)
    atoms_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return atoms_map, atoms


def checkmap(reactant_map, reactant_atom, product_map, product_atom):
    if 0 in product_map:
        print("has 0")
        return False

    if len(set(product_map)) != len(product_map):
        print("dup pro map")
        return False
    # for i in range(1, len(product_map) + 1):
    #     if i not in product_map:
    #         print("no order")
    #         return False
    reactant_map_tmp = [i for i in reactant_map if i != 0]
    if len(set(reactant_map_tmp)) != len(reactant_map_tmp):
        print("dup rea")
        return False

    a = [
        (reactant_map[i], reactant_atom[i])
        for i in range(len(reactant_map))
        if reactant_map[i] != 0
    ]
    b = [(product_map[i], product_atom[i]) for i in range(len(product_map))]

    a = set(a)
    b = set(b)
    # if not a >= b:
    if a != b:
        print("not match pro re1")
        return False

    reactant_map = set(reactant_map)
    product_map = set(product_map)

    if 0 in reactant_map:
        reactant_map.remove(0)
    # if not product_map <= reactant_map:
    if product_map != reactant_map:
        print("not match pro re2")
        return False
    return True


def get_product_and_reactant(smiles):
    tmp = smiles.split(">")
    reactant = tmp[0]
    product = tmp[2]
    return product, reactant


def check_smiles(product, reactant, len_=None):
    if error(reactant) is False or error(product) is False:
        print("error")
        return False
    # if check_edge(product, reactant) is False:
    #     return False
    reactant_map, reactant_atom = get_map(reactant)
    if len_ is not None:
        if len(reactant_atom) > len_:
            print("len", len(reactant_atom))
            return False
    product_map, product_atom = get_map(product)
    if len(product_atom) <= 1 or len(reactant_atom) <= 1:
        print("len(product_atom)")
        return False
    if "H" in reactant_atom or "H" in product_atom:
        print("has h")
        return False
    if checkmap(reactant_map, reactant_atom, product_map, product_atom) is False:
        return False
    return True


def check_lmdb(path, csv_path, len_=None):
    dataset = LMDBDataset(path)
    print(len(dataset))
    with open(csv_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "id", "rxn_smiles"])
        for i in tqdm(range(len(dataset))):
            result = dataset[i]
            class_ = result["class"]
            id = result["title"]
            rxn_smiles = result["ori_smiles"]
            rxn_smiles = rxn_smiles.replace(" ", "").split("|")[0]
            product, reactant = get_product_and_reactant(rxn_smiles)
            if check_smiles(product, reactant, len_=len_):
                writer.writerow([class_, id, rxn_smiles])
            # writer.writerows([[0,1,3],[1,2,3],[2,3,4]])


def check_csv(in_path, csv_path, len_=None):
    with open(csv_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "id", "rxn_smiles"])

        with open(in_path, "r", encoding="utf-8") as csvfile2:
            reader = csv.DictReader(csvfile2)
            for row in tqdm(reader):
                product, reactant = get_product_and_reactant(row["rxn_smiles"])
                if check_smiles(product, reactant, len_=len_):
                    writer.writerow([row["class"], row["id"], row["rxn_smiles"]])


def check_csv_2(path, csv_path, len_=None):
    with open(csv_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "id", "rxn_smiles"])

        with open(path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader):
                product, reactant = row["input"], row["target"]
                if check_smiles(product, reactant, len_=len_):
                    writer.writerow([0, 0, reactant + ">>" + product])
                # else:
                #     print(product, reactant)


def get_dictionary(f):
    with open(f, "r", encoding="utf-8") as fd:
        lines = [i.strip() for i in fd.readlines()]
    return set(lines)


def remove(path, csv_path, len_=None):
    with open(csv_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "id", "rxn_smiles"])

        with open(path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader):
                tmp = row["rxn_smiles"].split(">")
                reactant = tmp[0]
                reactant = get_atoms(reactant)
                len_rea = len(reactant)
                if len_ is None or len_rea <= len_:
                    writer.writerow([row["class"], row["id"], row["rxn_smiles"]])


def find_order(path1, dic_path):
    dic = get_dictionary(dic_path)
    smi_list = pd.read_csv(path1, names=["rxn_smiles"])["rxn_smiles"].tolist()  # [1:]
    print(smi_list[0])
    smi_list = smi_list[1:]
    set_ = set()
    dict_ = {}
    for i in tqdm(smi_list):
        tmp = i.split(">")
        reactant = tmp[0]
        product = tmp[2]
        reactant = get_atoms(reactant)
        len_rea = len(reactant)
        if len_rea in dict_.keys():
            dict_[len_rea] = dict_[len_rea] + 1
        else:
            dict_[len_rea] = 1

        # product = get_atoms(product)
    #     set_ = set_.union(reactant, product)
    # print(1, set_ - dic)
    dict_ = collections.OrderedDict(sorted(dict_.items()))
    for k, v in dict_.items():
        print(k, v)


if __name__ == "__main__":
    # path = "/data/users/zhen/dataset/retro_chem_data/reaction_map_num_db/{}.lmdb"
    # csv_path = "/data/users/yaolin/BioG2G_/400w_remove_raw_20230316/{}.csv"
    csv_path = "/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset_filter_all/{}.csv"

    for i in ["test", "valid", "train"]:
    # for i in ["uspto_testT", "uspto_valT", "uspto_trainT"]:
    # for i in ["test", "test1", "train.small", "valid", "valid1", "train"]:
        # check(path.format(i), csv_path.format(i), len_=150)
        # remove(
        #     path=csv_path.format(i),
        #     csv_path=csv_path.format(i).replace("raw", "raw_150"),
        #     len_=150,
        # )
        print(i, "*" * 30)
        check_csv(
            in_path=csv_path.format(i),
            csv_path=csv_path.format(i).replace("pistachio_clean_dataset_filter_all", "pistachio_clean_dataset_filter_200"),
            len_=200,
        )
        # find_order(csv_path.format(i), "/data/users/yaolin/BioG2G_/400w_raw/dict.txt")
