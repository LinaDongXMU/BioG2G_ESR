import random
from rdkit import Chem
import numpy as np
from torch.utils.data import Dataset
import copy


def get_random_smiles(smi, prob=1):
    if prob == 0 or random.random() >= prob:
        return smi
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    for i in range(5):
        smiles = Chem.MolToSmiles(mol, doRandom=True)
        if Chem.MolFromSmiles(smiles) is not None:
            return smiles
        print("RandomSmilesDataset doRandom Fail", i, smiles)
    raise Exception("Invalid smiles!")


class ReorderSmilesDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def get_map(self, smi):
        c_mol = Chem.MolFromSmiles(smi)
        c_id_list = [atom.GetAtomMapNum() for atom in c_mol.GetAtoms()]
        return c_id_list, c_mol

    def get_list(self, atoms_map_product, atoms_map_reactant):
        atoms_map_reactant_dict = {
            atoms_map_reactant[i]: i for i in range(len(atoms_map_reactant))
        }
        tmp = np.array([atoms_map_reactant_dict[i] for i in atoms_map_product])
        orders = np.array([i for i in range(len(atoms_map_reactant))])
        mask = np.array(atoms_map_reactant) != 0
        list_reactant = np.concatenate([tmp, orders[~mask]], 0).tolist()
        return list_reactant

    def get_new_smiles(self, product, reactant):
        product_map, _ = self.get_map(product)
        product_map = [i for i in product_map if i != 0]
        reactant_map, reactant_mol = self.get_map(reactant)
        list_reactant = self.get_list(product_map, reactant_map)
        nm = Chem.RenumberAtoms(reactant_mol, list_reactant)
        new_smiles = Chem.MolToSmiles(nm, canonical=False)
        return new_smiles, nm

    def __getitem__(self, index: int):
        product = self.dataset[index]["product_smiles"]
        reactant = self.dataset[index]["reactant_smiles"]
        new_smiles, _ = self.get_new_smiles(product, reactant)
        new_smiles, new_mol = self.get_new_smiles(product, new_smiles)
        if Chem.MolFromSmiles(new_smiles) is None:
            print("ReorderSmilesDataset Fail", reactant, new_smiles)
        # the new_mol is the reordered graph
        return {"mol": new_mol, "smiles": new_smiles}


def rearrange_and_pad(arr, index_list, pad_value=0):
    if len(arr) == len(index_list):
        return arr[index_list]
    result = np.array([arr[idx] if idx != -1 else pad_value for idx in index_list])
    return result


def shuffle_graph_process(result, list_):
    result_keys = result.keys()
    for i in [
        ("atoms", "NoneAtom"),
        ("atoms_map", 0),
        ("node_attr", np.array([0 for _ in range(9)])),
        ("atoms_token", 0),
    ]:
        key, pad = i
        if key in result_keys:
            result[key] = rearrange_and_pad(result[key], list_, pad_value=pad)
    list_reverse = {i: idx for idx, i in enumerate(list_)}
    for i in range(result["edge_index"].shape[0]):
        for j in range(result["edge_index"].shape[1]):
            result["edge_index"][i, j] = list_reverse[result["edge_index"][i, j]]
    return result


class ReorderGraphDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def get_list(self, atoms_map_product, atoms_map_reactant):
        assert 0 not in atoms_map_product
        atoms_map_reactant_dict = {
            atoms_map_reactant[i]: i for i in range(len(atoms_map_reactant))
        }
        tmp = [atoms_map_reactant_dict[i] if i != 0 else -1 for i in atoms_map_product]
        orders = np.array([i for i in range(len(atoms_map_reactant))])
        mask = atoms_map_reactant != 0
        list_reactant = np.concatenate([tmp, orders[~mask]], 0)
        return None, list_reactant

    def __getitem__(self, index: int):
        product = self.dataset[index]["product"]
        reactant = self.dataset[index]["reactant"]
        try:
            list_product, list_reactant = self.get_list(
                product["atoms_map"], reactant["atoms_map"]
            )
        except:
            raise
        if list_product is not None:
            product = shuffle_graph_process(product, list_=list_product)
        if list_reactant is not None:
            reactant = shuffle_graph_process(reactant, list_=list_reactant)
        return {"reactant": reactant, "product": product}


allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    # https://www.rdkit.org/docs/cppapi/classRDKit_1_1Atom.html
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "misc",
    ],
    "possible_degree_list": list(range(31)) + ["misc"],
    # change
    "possible_formal_charge_list": [i - 31 // 2 for i in range(31)] + ["misc"],
    "possible_numH_list": list(range(31)) + ["misc"],
    "possible_number_radical_e_list": list(range(31)) + ["misc"],
    # https://www.rdkit.org/docs/cppapi/classRDKit_1_1Atom.html#a58e40e30db6b42826243163175cac976
    "possible_hybridization_list": [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "S",
        "SP2D",
        "OTHER",
        "UNSPECIFIED",
        "misc",
    ],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    # https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html#a2c93af0aeb3297ee77b6afdc27b68d6f
    "possible_bond_type_list": [
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
        "UNSPECIFIED",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
        "misc",
    ],
    # https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html#ae91dd8e72b495a48f46775c874882165
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
    "possible_bond_dir_list": [
        "NONE",
        "BEGINWEDGE",
        "BEGINDASH",
        "ENDDOWNRIGHT",
        "ENDUPRIGHT",
        "EITHERDOUBLE",
        "R_BEGINWEDGE",
        "R_BEGINDASH",
        "R_ENDDOWNRIGHT",
        "R_ENDUPRIGHT",
        "R_EITHERDOUBLE",
        "UNKNOWN",
    ],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        print(l, e)
        # raise
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
        safe_index(
            allowable_features["possible_bond_dir_list"], str(bond.GetBondDir())
        ),
    ]
    return bond_feature


def get_graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    # bonds
    num_bond_features = 4  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feature2 = copy.deepcopy(edge_feature)
            if edge_feature2[3] > 0 and edge_feature2[3] < 6:
                edge_feature2[3] += 5
            edge_features_list.append(edge_feature2)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int32).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)
    return x, edge_index, edge_attr


def process_one(smiles):
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    atoms = np.array([x.GetSymbol() for x in mol.GetAtoms()])
    atoms_map = np.array([x.GetAtomMapNum() for x in mol.GetAtoms()])
    node_attr, edge_index, edge_attr = get_graph(mol)
    return {
        "atoms": atoms,
        "atoms_map": atoms_map,
        "smi": smiles,
        "node_attr": node_attr,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }


class Mol2GraphDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index: int):
        smiles_or_mol = self.dataset[index]
        try:
            result = process_one(smiles_or_mol)
        except:
            print("process_one", smiles_or_mol)
            raise
        return result


if __name__ ==  "__main__":
    # simple example:
    reactant = "O=C(OCc1ccccc1)[NH:10][CH2:9][CH2:8][CH2:7][CH2:6][C@@H:5]([C:3]([O:2][CH3:1])=[O:4])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]"
    init_product = "[CH3:1][O:2][C:3](=[O:4])[C@H:5]([CH2:6][CH2:7][CH2:8][CH2:9][NH2:10])[NH:11][C:12](=[O:13])[NH:14][c:15]1[cH:16][c:17]([O:18][CH3:19])[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[OH:27]"

    for _ in range(4):
        print("*"*80)
        # get a product with random atom order
        product = get_random_smiles(init_product)
        reorder_smiles_dataset = ReorderSmilesDataset(dataset=[{"reactant_smiles":reactant, "product_smiles":product}])

        reactant_aligned = reorder_smiles_dataset[0]
        reactant_aligned_smiles = reorder_smiles_dataset[0]["smiles"]
        reactant_aligned_graph_dataset = Mol2GraphDataset(dataset=[reactant_aligned["mol"]])
        product_graph_dataset = Mol2GraphDataset(dataset=[product])


        print("------------product_graph------------")
        print(product)
        for k, v in product_graph_dataset[0].items():
            if k in ["atoms", "atoms_map"]:
                print(k, v)
        print("------------reactant_aligned_graph------------")
        print(reactant_aligned_smiles)
        for k, v in reactant_aligned_graph_dataset[0].items():
            if k in ["atoms", "atoms_map"]:
                print(k, v)

