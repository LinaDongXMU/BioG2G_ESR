from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

try:
    from rdkit.Chem import Draw
except:
    print("can not import chem draw")
from itertools import product
from copy import deepcopy
from collections import OrderedDict

# from basic import draw_mol

np.set_printoptions(threshold=np.inf)

flag_kekulize = False
flag_atoms_chiraltag = "new"
flag_use_list = False

# 22 type
bond_type_list = [
    Chem.rdchem.BondType.UNSPECIFIED,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.QUADRUPLE,
    Chem.rdchem.BondType.QUINTUPLE,
    Chem.rdchem.BondType.HEXTUPLE,
    Chem.rdchem.BondType.ONEANDAHALF,
    Chem.rdchem.BondType.TWOANDAHALF,
    Chem.rdchem.BondType.THREEANDAHALF,
    Chem.rdchem.BondType.FOURANDAHALF,
    Chem.rdchem.BondType.FIVEANDAHALF,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.IONIC,
    Chem.rdchem.BondType.HYDROGEN,
    Chem.rdchem.BondType.THREECENTER,
    Chem.rdchem.BondType.DATIVEONE,
    Chem.rdchem.BondType.DATIVE,
    Chem.rdchem.BondType.DATIVEL,
    Chem.rdchem.BondType.DATIVER,
    Chem.rdchem.BondType.OTHER,
    Chem.rdchem.BondType.ZERO,
]

chiral_type_list_1 = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,  # chirality that hasn't been specified
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,  # tetrahedral: clockwise rotation (SMILES @@)
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,  # tetrahedral: counter-clockwise rotation (SMILES @)
    Chem.rdchem.ChiralType.CHI_OTHER,  # some unrecognized type of chirality
    # Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,  # tetrahedral, use permutation flag
    # Chem.rdchem.ChiralType.CHI_ALLENE,  # allene, use permutation flag
    # Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,  # square planar, use permutation flag
    # Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,  # trigonal bipyramidal, use permutation flag
    # Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,  # octahedral, use permutation flag
]

chiral_type_list = ["", "S", "R"]

bond_stereo_list = [  # stereochemistry of double bonds
    Chem.rdchem.BondStereo.STEREONONE,  # no special style
    Chem.rdchem.BondStereo.STEREOANY,  # intentionally unspecified
    # -- Put any true specifications about this point so
    # that we can do comparisons like if(bond->getStereo()>Bond::STEREOANY)
    Chem.rdchem.BondStereo.STEREOZ,  # Z double bond
    Chem.rdchem.BondStereo.STEREOE,  # E double bond
    Chem.rdchem.BondStereo.STEREOCIS,  # cis double bond
    Chem.rdchem.BondStereo.STEREOTRANS,  # trans double bond
]


def set_h_number(mol, atom_h_number):
    for i in range(len(atom_h_number)):
        for _ in range(atom_h_number[i]):
            atom_tmp = Chem.Atom("H")
            molecular_index = mol.AddAtom(atom_tmp)
            try:
                mol.AddBond(i, molecular_index, Chem.rdchem.BondType.SINGLE)
            except:
                mol.RemoveAtom(molecular_index)


def get_adjacency_matrix(smiles, add_h=None):
    mol = Chem.MolFromSmiles(smiles)
    if flag_kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    if add_h is True:
        mol = AllChem.AddHs(mol)
    elif add_h is False:
        mol = AllChem.RemoveHs(mol)
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    bond_stereo = np.zeros_like(adjacency_matrix)
    bond_stereo_dict = dict()
    for bond in mol.GetBonds():
        adjacency_matrix[
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ] = bond_type_list.index(bond.GetBondType())
        adjacency_matrix[
            bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()
        ] = bond_type_list.index(bond.GetBondType())
        bond_stereo[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetStereo()
        bond_stereo[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetStereo()
        stereo_tmp = list(bond.GetStereoAtoms())
        if len(stereo_tmp) >= 2:
            bond_stereo_dict[
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            ] = stereo_tmp
            bond_stereo_dict[
                (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            ] = stereo_tmp
    atoms_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    atoms_charge = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_h_number = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
    if flag_atoms_chiraltag == "old":
        atoms_chiraltag = [
            chiral_type_list_1.index(atom.GetChiralTag()) for atom in mol.GetAtoms()
        ]
    else:
        atoms_chiraltag = [0 for _ in range(mol.GetNumAtoms())]
        for i, c in Chem.FindMolChiralCenters(mol):
            atoms_chiraltag[i] = chiral_type_list.index(c)
    return {
        "adjacency_matrix": adjacency_matrix,
        "atoms": atoms,
        "atoms_map": atoms_map,
        "atoms_chiraltag": atoms_chiraltag,
        "atoms_charge": atoms_charge,
        "bond_stereo": bond_stereo,
        "bond_stereo_dict": bond_stereo_dict,
        "atom_h_number": atom_h_number,
    }


def graph2mol(
    adjacency_matrix,
    atoms,
    atoms_map=None,
    atoms_chiraltag=None,
    atoms_charge=None,
    bond_stereo=None,
    bond_stereo_dict=None,
    atom_h_number=None,
    add_h=False,
):
    molecule = Chem.RWMol()
    atom_index = []
    for atom_number in range(len(atoms)):
        atom_tmp = Chem.Atom(atoms[atom_number])
        if atoms_map is not None:
            atom_tmp.SetAtomMapNum(atoms_map[atom_number])
        if atoms_charge is not None:
            atom_tmp.SetFormalCharge(atoms_charge[atom_number])
        if atoms_chiraltag is not None and flag_atoms_chiraltag == "old":
            atom_tmp.SetChiralTag(chiral_type_list_1[atoms_chiraltag[atom_number]])

        molecular_index = molecule.AddAtom(atom_tmp)
        atom_index.append(molecular_index)
    for index_x, row_vector in enumerate(adjacency_matrix):
        for index_y, bond in enumerate(row_vector):
            if index_y <= index_x:
                continue
            if bond == 0:
                continue
            else:
                molecule.AddBond(
                    atom_index[index_x], atom_index[index_y], bond_type_list[bond]
                )

    if bond_stereo is not None:
        for bond in molecule.GetBonds():
            stereo_tmp = bond_stereo[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            bond.SetStereo(bond_stereo_list[stereo_tmp])
            key = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if key in bond_stereo_dict.keys():
                tmp = bond_stereo_dict[key]
                bond.SetStereoAtoms(tmp[0], tmp[1])
    if atom_h_number is not None:
        set_h_number(molecule, atom_h_number)
    molecule = molecule.GetMol()
    try:
        molecule.UpdatePropertyCache()
    except:
        pass
    if atoms_chiraltag is not None and flag_atoms_chiraltag == "new":
        trials = [
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        ]
        chis = OrderedDict()
        for i, c in enumerate(atoms_chiraltag):
            atIdx = atom_index[i]
            if c > 0:
                chis.update({atIdx: chiral_type_list[c]})

        if len(chis) > 0:
            if flag_use_list:
                molecule_list = []
            for prod in product(trials, repeat=len(chis)):
                m = deepcopy(molecule)

                for i, atIdx in enumerate(chis.keys()):
                    m.GetAtomWithIdx(atIdx).SetChiralTag(prod[i])

                matches = []
                Chem.AssignStereochemistry(m)
                for atIdx, c in Chem.FindMolChiralCenters(m):
                    matches.append(chis[atIdx] == c)
                # print(prod, matches)
                if all(matches):
                    if flag_use_list:
                        molecule_list.append(m)
                    else:
                        molecule = m
                        break
            if flag_use_list:
                molecule = molecule_list
        else:
            Chem.AssignStereochemistry(molecule)
            if flag_use_list:
                molecule = [molecule]

    # Chem.AssignAtomChiralTagsFromStructure(molecule)
    # molecule = AllChem.RemoveHs(molecule)
    # return molecule
    if flag_kekulize:
        smiles = Chem.MolToSmiles(molecule, kekuleSmiles=True)
    else:
        # molecule = AllChem.RemoveHs(molecule)
        if flag_use_list:
            # smiles = [Chem.MolToSmiles(AllChem.RemoveHs(i)) for i in molecule]
            if isinstance(molecule, list):
                molecule = [molecule]
            if add_h:
                molecule = [AllChem.AddHs(i) for i in molecule]
            smiles = [Chem.MolToSmiles(i) for i in molecule]
        else:
            # molecule = AllChem.RemoveHs(molecule)
            if add_h:
                molecule = AllChem.AddHs(molecule)
            smiles = Chem.MolToSmiles(molecule)
    return smiles


def get_InchiKey(smi):
    if smi is None or smi == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return None
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
        return key
    except:
        return None


def judge_InchiKey(key1, key2):
    if key1 is None or key2 is None:
        return False
    return key1 == key2


def same_smi(smi1, smi2):
    key1 = get_InchiKey(smi1)
    if key1 is None:
        return False
    key2 = get_InchiKey(smi2)
    if key2 is None:
        return False
    return judge_InchiKey(key1, key2)


def get_charge_dict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms_charge = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    atoms_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    dict_ = {atoms_map[i]: atoms_charge[i] for i in range(len(atoms_charge))}
    dict_ = dict(sorted(dict_.items()))
    # print(dict_)
    if 0 in dict_.keys():
        dict_.pop(0)
    return dict_


def get_dict(path):
    with open(path, "r") as f:
        a = [i.strip() for i in f.readlines()]
    return a


def test(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.RemoveHs(mol)
    # chis1 = list(Chem.FindMolChiralCenters(mol))
    # print([bo.GetStereo() for bo in mol.GetBonds()])

    # Draw.MolToFile(mol, "test1.png", (1000, 1000))

    if flag_kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        smiles_refined = Chem.MolToSmiles(mol, kekuleSmiles=True)
    else:
        if False:
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
            smiles_refined = Chem.MolToSmiles(mol)
        else:
            smiles_refined = smiles
    kwargs = get_adjacency_matrix(smiles_refined, add_h=None)

    smiles2 = graph2mol(**kwargs)
    # chis2 = list(Chem.FindMolChiralCenters(mol))
    # print([tuple(bo.GetStereoAtoms()) for bo in mol.GetBonds()])
    # Draw.MolToFile(mol, "test2.png", (1000, 1000))
    if flag_use_list:
        same = [same_smi(smiles_refined, i) for i in smiles2]
        same = True if True in same else False
    else:
        same = same_smi(smiles_refined, smiles2)
    if not same:
        print("*" * 10)
        print(smiles)
        print(smiles_refined)
        print(smiles2)
        # print(chis1, chis2)
    # draw_mol([smiles, smiles_refined, smiles2], "2.png")
    return same


def test2(smiles, lis):
    mol = Chem.MolFromSmiles(smiles)
    if flag_kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for i in atoms:
        if i not in lis:
            print(i)
            return False
    return True


def test4(smiles, all_reactant_atoms):
    mol = Chem.MolFromSmiles(smiles)
    if flag_kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atoms_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    for i in range(len(atoms_map)):
        if atoms_map[i] == 0:
            if atoms[i] in all_reactant_atoms.keys():
                all_reactant_atoms[atoms[i]] += 1
                all_reactant_atoms["all"] += 1

            else:
                all_reactant_atoms[atoms[i]] = 1
                all_reactant_atoms["all"] += 1

    return True


def test3(smiles):
    kwargs = get_adjacency_matrix(smiles, add_h=None)
    # adjacency_matrix, atoms, atoms_charge, atom_h_number
    list_ = ["atoms_map", "atoms_chiraltag", "bond_stereo", "bond_stereo_dict"]
    for i in list_:
        kwargs.pop(i)
    smiles2 = graph2mol(**kwargs)
    same = smiles2 is not None
    return same


if __name__ == "__main__":
    smiles = [
        # "[CH3:1][CH2:2][C:3](=[O:4])[O:5][C@H:6]1[CH2:7][CH2:8][C@H:9]2[C@@H:10]3[CH2:11][CH2:12][C:13]4=[CH:14][C:15](=[O:16])[CH2:17][CH2:18][C@:19]4([CH2:20][OH:21])[C@H:22]3[CH2:23][CH2:24][C@:25]12[CH3:26]",
        # "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH:6]([CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH:12]=[CH:13][CH2:14][C@H:15]1[c:16]2[cH:17][cH:18][c:19]([O:20][CH3:21])[cH:22][c:23]2[S:24][CH2:25][C@@:26]1([CH3:27])[c:28]1[cH:29][cH:30][c:31]([O:32][CH3:33])[cH:34][cH:35]1)[CH2:36][CH2:37][CH2:38][C:39]([F:40])([F:41])[C:42]([F:43])([F:44])[F:45]",
        # "[O:1]=[C:2]([OH:3])[CH2:4][C@@H:5]1[CH2:6][c:7]2[cH:8][c:9]([Br:10])[c:11]3[nH:12][n:13][c:14]([Cl:15])[c:16]3[c:17]2[CH2:18][N:19]([CH2:20][C:21]([F:22])([F:23])[F:24])[C:25]1=[O:26]"
        "[H]C([H])([H])C(OC=O)(C([H])([H])[H])C([H])([H])[H].[H]C([H])([H])ON(C(=O)C([H])([H])([H])[Mg+])C([H])([H])[H].[H]c1cc([H])c2c([H])c([H])nc2c1[H]",
        "Br.[H]C(#Cc1c([H])c([H])c(C([H])([H])C([H])(N([H])C(=O)C([H])([H])[H])C([H])([H])[H])c([H])c1[H])c1c([H])c([H])c(OC([H])([H])[H])c([H])c1OC([H])([H])[H]",
        "C.[H]C([H])[H].[H]C([H])[H].[H]C([H])[H].[H]c1nn([H])c(C([H])([H])[H])c1-c1(C([H])([H])[H])sc2c(=O)n([H])c([C@]3([H])N(C(=O)OCC([H])([H])[H])C4([H])C([H])([H])C([H])([H])C3([H])C([H])([H])C4([H])[H])nc2c1[H]",
    ]
    # for i in smiles:
    #     mol = Chem.MolFromSmiles(i)
    # global draw_count
    # draw_mol([i], "img3/{}.png".format(draw_count))
    # draw_count = draw_count + 1
    import pandas as pd
    from tqdm import tqdm

    # a = get_dict(
    # "/data/users/yaolin/BioG2G_/retro/USPTO50K_ALL_20221220_1_rmh/encoder_dict2.txt"
    # )
    df = pd.read_csv("/data/users/yaolin/BioG2G_/retro/reactant/USPTO50K_raw.csv")
    smiles = tqdm(list(df["smi"]))
    flag = 0
    all_reactant_atoms = {"all": 0}
    for smi in smiles:
        # test4(smi, all_reactant_atoms)
        if not test(smi):
            flag += 1
    print(flag)
    # print(dict_)
    # all_reactant_atoms = dict(sorted(all_reactant_atoms.items(), key=lambda item: item[1],reverse=True))
    # print(all_reactant_atoms)
    # for k, v in all_reactant_atoms.items():
    #     print(k)
    #     print(v / all_reactant_atoms["all"])
