import sys
sys.path.append("/data/users/yaolin/BioG2G/BioG2G_model")
from data_preprocess.basic import brief_dataset, LMDBDataset
from rdkit import Chem
from tqdm import tqdm
import csv

def process_atom_maps(smiles_a, smiles_b):
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    
    atom_maps_a = [atom.GetAtomMapNum() for atom in mol_a.GetAtoms()]
    atom_maps_b = [atom.GetAtomMapNum() for atom in mol_b.GetAtoms()]
    
    missing_maps = set(atom_maps_b) - set(atom_maps_a + [0])
    
    for atom in mol_b.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map in missing_maps:
            atom.SetAtomMapNum(0)
    
    smiles_b = Chem.MolToSmiles(mol_b)
    return smiles_b

def canonical(testsmi):
    # print(testsmi)
    if testsmi == "" or testsmi is None:
        return ""
    mol = Chem.MolFromSmiles(testsmi)
    if mol is None:
        print("no mol", testsmi)
        return ""
    canonical_smi = Chem.MolToSmiles(mol)
    if canonical_smi == "" or canonical_smi is None:
        print("no canonical_smi", testsmi)
        return ""
    return canonical_smi

def lmdb2csv(path, outfile):
    if not isinstance(path, list):
        path = [path]

    header = ["class", "id", "rxn_smiles", "ori_smiles", "map_type", "map_score"]
    with open(outfile, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        count = 0
        type_list = type([])
        type_str = type("34")
        for p in path:
            dataset = LMDBDataset(p)
            print(len(dataset))
            for i in tqdm(range(len(dataset))):
                list_ = [0]
                result = dataset[i].to_dict()
                # result = dataset[i]
                list_.append(result.pop("title", ""))
                assert type(result["smiles_mapnumber_reactant_list"]) == type_list
                reactant = ".".join(result["smiles_mapnumber_reactant_list"])
                reactant = canonical(reactant)
                if reactant == "":
                    count += 1
                    print(count)
                    continue
                    raise
                precursor = ".".join(result["smiles_mapnumber_precursor_list"])
                # precursor = canonical(precursor)
                # if precursor == "":
                #     raise
                target = result["smiles_mapnumber_target_list"]
                if type(target) == type_str:
                    target = [target]
                if True:
                    if len(target) != 1:
                        print("len target", len(target))
                        print("len target", target)
                        count += 1
                        print(count)
                        continue
                        raise
                    assert "." not in target
                    target = canonical(target[0])
                    if target == "":
                        count += 1
                        print(count)
                        continue
                        raise
                else:
                    target = ".".join(target)
                    target = canonical(target)
                    if target == "":
                        count += 1
                        print(count)
                        continue
                        raise
                try:
                    reactant = process_atom_maps(target, reactant)
                except:
                    print(reactant)
                    print(target)
                    count += 1
                    print(count)
                    continue
                list_.append(reactant + ">" + precursor + ">" + target)
                list_.append(result["ori_smiles"])
                list_.append(result.pop("map_type", ""))
                list_.append(result.pop("map_score",""))
                writer.writerow(list_)


        print(count)

if __name__ == "__main__":
    p="/data/users/zhen/dataset/retro_chem_data/pistachio_2023/pistachio_clean_split_pro/train.lmdb"
    outfile = "/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset/train.csv"
    # brief_dataset(p)
    lmdb2csv(p, outfile)
