from unimol import __version__
from unicore import distributed_utils
# from BioG2G.utils import save_config

if __version__ in ["1.5.0", "2.0.0", "2.5.0"]:
    import logging
    import os
    import torch
    import numpy as np
    from unicore.data import (
        Dictionary,
        LMDBDataset,
        RightPadDataset,
        # TokenizeDataset,
        RightPadDataset2D,
        NestedDictionaryDataset,
        EpochShuffleDataset,
    )
    from unimol.data import (
        KeyDataset,
    )
    # from BioG2G.data import (
    #     CsvGraphormerDataset,
    #     SmilesDataset,
    #     GraphormerDataset,
    #     SeqGraphormerDataset,
    #     RightPadDataset3D,
    #     ReorderGraphormerDataset,
    #     GraphFeatures,
    #     RandomSmilesDataset,
    #     ReorderSmilesDataset,
    #     EMPTY_SMILES_Dataset_G2GT,
    #     SmilesDataset_2,
    #     BpeTokenizeDataset,
    #     TokenizeDataset,
    # )
    # from BioG2G.utils.chemutils import add_chirality
    # from BioG2G.utils.G2GT_cal import get_smiles, gen_map
    from unicore.tasks import UnicoreTask, register_task
    from unicore import checkpoint_utils
from rdkit import Chem
from tqdm import tqdm
import json


def smi2smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    smiles = Chem.MolToSmiles(mol)
    return smiles


src_path = '/data/projects/unimol/unimol_v1_data/examples/unimol_v2'
dest_path = '/data/users/qifanyu/nag2g_BioG2G/BioG2G_/unimol_v1_data'
subsets = [subset.split('.lmdb')[0] for subset in sorted(
    os.listdir(src_path)) if subset.endswith('train.lmdb')]
# print(subsets)


for subset in subsets:
    split_path = os.path.join(src_path, subset + '.lmdb')
    raw_dataset = LMDBDataset(split_path)
    # print(raw_dataset[0]['smi'])
    dataset = [
        smi2smiles(sample['smi']) for sample in tqdm(raw_dataset)
    ]
    # print(dataset[:10])

    filename = os.path.join(dest_path, subset + '.json')
    with open(filename, 'w') as fp:
        json.dump(dataset, fp, indent=4)

    print(filename)
