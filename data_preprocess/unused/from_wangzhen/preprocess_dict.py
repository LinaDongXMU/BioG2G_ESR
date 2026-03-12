import timeout_decorator
import argparse
from multiprocessing import Pool
import contextlib
import warnings
import os
import sys
import json
import glob
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger
from random import shuffle
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@timeout_decorator.timeout(4)
def inner_process(smi):
    try:
        reg = smi['rec'].split()
        tar = smi['pro'].split()
        size_reg = smi['len_rec']
        size_tar = smi['len_pro']
        temperate_cat = smi['temp_cat']
        pressure_cat = smi['pres_cat']

        clas = process_class(smi['class'])
        # print('test reg and tar: ', reg, tar, size_reg, size_tar, temperate_cat, pressure_cat, clas)
        return pickle.dumps({'smiles_train': reg, 'smiles_target': tar, 'rec_size': size_reg, 'tar_size': size_tar,
                             'temperate_cat': temperate_cat, 'pressure_cat': pressure_cat, 'class': clas, }, protocol=-1)
    except:
        print('inner fail: ', smi)
        return None


def process_class(clas):
    clas_list = clas.split('.')
    if len(clas_list) == 3:
        return clas
    elif len(clas_list) == 2:
        return clas_list[0]+'.'+clas_list[1]+'.0'
    elif len(clas_list) == 1:
        return clas_list[0]+'.0.0'
    else:
        return '0.0.0'


def data_process(smi):
    try:
        return inner_process(smi)
    except:
        print('fail: ', smi)
        return None


def get_train(train_file):
    reaction_data = np.load(train_file, allow_pickle=True)
    train_data = []
    val_data = []
    test_data = []
    for i, item in enumerate(reaction_data):
        if item['training_set'] == 0:
            train_data.append(item)
        elif item['training_set'] == 1:
            val_data.append(item)
        else:
            test_data.append(item)
        # if i > 1500: break
    return train_data, val_data, test_data


def write_lmdb(inputfilename1, outpath='.', nthreads=16):

    # small_size = 10000
    train_data, val_data, test_data = get_train(inputfilename1)
    train_data_demo = train_data[:1000]
    print('test train_smi: ', len(train_data))
    for name, smi_list in [('train.lmdb', train_data), ('train_demo.lmdb', train_data_demo), ('valid.lmdb', val_data), ('test.lmdb', test_data)]:
        outputfilename = os.path.join(outpath, name)
        # print('outputfilename: ', outputfilename)
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
        # with Pool(nthreads) as pool:
        i = 0
        smi_list_l = [data_process(smi) for smi in tqdm(smi_list)]
        print('len smi: ', len(smi_list_l))
        # for inner_output in tqdm(pool.imap(smiprocess, smi_list), total=len(smi_list)):
        for inner_output in smi_list_l:
            if inner_output is not None:
                txn_write.put(f'{i}'.encode("ascii"), inner_output)
                i += 1
                if i % 10000 == 0:
                    txn_write.commit()
                    txn_write = env_new.begin(write=True)
        # print('{} process {} lines'.format(name, i))
        txn_write.commit()
        print('{} process {} lines final'.format(name, i))
        env_new.close()


def connect_db(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    return env


def check_lmdb(data_path):

    env = connect_db(data_path)
    with env.begin() as txn:
        i = 0
        for key, value in tqdm(txn.cursor()):
            data = pickle.loads(value)

            # print('test: ', key.decode('utf8'))
            i += 1
        print('{} process {} lines'.format(key, i))


def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-in', '--inputfile1',
                        default='/mnt/vepfs/users/zhen/dataset/retro_chem_data/reaction_data_spilt.npy')
    parser.add_argument('-out', '--output_path',
                        default='/mnt/vepfs/users/zhen/dataset/retro_chem_data/stand_mol_data_double_test')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser()
    write_lmdb(args.inputfile1, args.output_path, nthreads=16)

    # data_path = '/mnt/vepfs/zhen/retro_analysis/molecular_transformer/dataset/MIT_MIXED_STAND/test.lmdb'
    # check_lmdb(data_path)
