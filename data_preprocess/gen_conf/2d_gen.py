import os
import sys
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool

def smi2scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smi, includeChirality=True)
    except:
        print("failed to generate scaffold with smiles: {}".format(smi))
        return smi

def csv_file_read(path, usecols=None):
    head_row = pd.read_csv(path, nrows=0)
    print(list(head_row))
    head_row_list = list(head_row)
    if usecols is None:
        usecols=head_row_list
    csv_result = pd.read_csv(path, usecols=usecols)
    row_list = csv_result.values.tolist()
    return row_list

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates

def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def inner_smi2coords(content):
    smi = content
    mol = Chem.MolFromSmiles(smi)
    coordinate_list=[smi2_2Dcoords(smi).astype(np.float32)]
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol':mol,'smi': smi}, protocol=-1)

def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None

def write_lmdb(file_in="./clean_smi.csv.gz", outfile=None, outpath='.', nthreads=16, range=None):

    small_size = 10000
    # smi_list = pd.read_csv(file_in, names=['smi'], #nrows=100000
    #                        )['smi'].tolist()
    # smi_list = csv_file_read(file_in, ["input"])
    # smi_list = [i[0] for i in smi_list]
    smi_list = csv_file_read(file_in, ["rxn_smiles"])
    smi_list = [i[0].split(">")[-1] for i in smi_list]
    print(smi_list[0])
    smi_list = smi_list[1:]
    # smi_list = smi_list[range[0]:range[1]]
    print('original size: ', len(smi_list))
    # train_smi, val_smi = get_train_val(smi_list, val_size=100000)
    # print('train size: {}; val size: {}'.format(len(train_smi), len(val_smi)))
    task_list = [
        # ('valid.lmdb', val_smi),
        # ('train.small.lmdb', train_smi[:small_size]),
        # ('train.lmdb', train_smi),
        (outfile, smi_list),

    ]
    for name, smi_list in task_list:
        outputfilename = os.path.join(outpath, name)
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
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(smi2coords, smi_list), total=len(smi_list)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
                    if i % 10000 == 0:
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()

if __name__ == '__main__':
    # write_lmdb(file_in="/data/projects/BioG2G/dataset/synthesis/uspto-full/extract/uspto_trainT.csv",
    #            outfile="uspto_trainT_2d.lmdb", outpath='/data/projects/BioG2G/dataset/synthesis/uspto-full/extract/', nthreads=5, range=None)


    write_lmdb(
        file_in="/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset_filter_all/test.csv",
        outfile="test_2d.lmdb",
        outpath="/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset_filter_all/",
        nthreads=15,
        # range=(0, 10),
    )
    write_lmdb(
        file_in="/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset_filter_all/valid.csv",
        outfile="valid_2d.lmdb",
        outpath="/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset_filter_all/",
        nthreads=15,
        # range=(0, 10),
    )
    write_lmdb(
        file_in="/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset_filter_all/train.csv",
        outfile="train_2d.lmdb",
        outpath="/data/users/yaolin/BioG2G_/pistachio/pistachio_clean_dataset_filter_all/",
        nthreads=15,
        # range=(0, 10),
    )