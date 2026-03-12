import sys
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from draw_img import draw_mol
from chemutils import add_chirality
from graph_process import seq2graph
from mol_graph_basic import graph2mol, error, get_InchiKey, judge_InchiKey, same_smi
import os
from multiprocessing import Pool
import time
import glob
import json

def seq2smiles(seq, atom_map=None): #输入图的seq token，输出smiles
    tmp = seq2graph(seq)
    adjacency_matrix = tmp["adj_matrix"] + 1
    adjacency_matrix[adjacency_matrix == 4] = 12
    adjacency_matrix[adjacency_matrix == 23] = 0
    atoms = tmp["atoms"]
    atoms_charge = tmp["atoms_charge"].tolist()
    atom_h_number = tmp["atoms_h"].tolist()
    if atom_map is not None:
        if len(atom_map) > len(atoms):
            atom_map = atom_map[: len(atoms)]
        elif len(atom_map) < len(atoms):
            atom_map = atom_map + [0 for _ in range(len(atoms) - len(atom_map))]

    smiles = graph2mol(
        adjacency_matrix=adjacency_matrix,
        atoms=atoms,
        atoms_charge=atoms_charge,
        atom_h_number=atom_h_number,
        atoms_map=atom_map,
    )
    return smiles


def get_smiles(seq, atom_map=None): #调用了上面的seq2smiles，进一步处理了seq里的特殊字符比如[SEP]
    seq = [i for i in seq.split(" ") if i != ""]
    if "[SEP]" in seq:
        seq = seq[: seq.index("[SEP]")]
    smiles = seq2smiles(seq, atom_map)
    return smiles


def gen_map(smiles): #输入smiles（猜测是带map的），输出atoms_map列表
    mol = Chem.MolFromSmiles(smiles)
    atoms_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    return atoms_map


def worker(args): #输入的参数包括i, context, iter_N, N_beam_search，输出的是smiles_infer.txt里的一个字典条目
    i, context, iter_N, N_beam_search = args
    gt_product = context[iter_N * i].split("gt_product")[0]
    target1 = context[iter_N * i + 1].split("target")[0]
    target2 = context[iter_N * i + 2].replace("[SEP]", "").split("target")[0]
    target2 = get_smiles(target2)

    dup_key_list = set([None])
    gt_reactant_key = get_InchiKey(target1)

    pred_list = []
    unstrict_list = []
    strict_list = []
    nodup_list = []
    pred_nodup_list = []
    flag_strict = False
    flag_unstrict = False
    for j in range(N_beam_search):
        assert "predicted" in context[iter_N * i + 3 + j]
        str2 = context[iter_N * i + 3 + j].split("predicted")[0]
        pred = get_smiles(str2)

        if flag_unstrict is False and same_smi(target2, pred):
            unstrict_list.append(1)
            flag_unstrict = True
        else:
            unstrict_list.append(0)

        pred = get_smiles(str2, atom_map=gen_map(gt_product))
        try:
            pred = add_chirality(gt_product, pred)
        except:
            pass

        pred_key = get_InchiKey(pred)
        if pred_key not in dup_key_list:
            if judge_InchiKey(pred_key, gt_reactant_key):
                nodup_list.append(1)
            else:
                nodup_list.append(0)
            dup_key_list.add(pred_key)
            pred_nodup_list.append(pred)
        if flag_strict is False and same_smi(target1, pred):
            strict_list.append(1)
            flag_strict = True
        else:
            strict_list.append(0)

        pred_list.append(pred)

    nodup_list = nodup_list + [0 for _ in range(N_beam_search - len(nodup_list))]

    # print(i, "*"*10)
    # print(target1, error(target1))
    # print(i, "*"*10)
    # for ii in pred_nodup_list:
    #     print(ii, error(ii))
    # if np.array(strict_list).sum() == 0:
    #     draw_mol(
    #         [gt_product, target1, None, None, None] + pred_nodup_list,
    #         "img/{}_{}.png".format(i, j),
    #         mols_per_row=5,
    #         img_size=(400, 400),
    #     )
    return {"product": gt_product, 
            "target": [target1, target2], 
            "unstrict_list": unstrict_list, 
            "strict_list": strict_list, 
            "pred_list": pred_list, 
            "nodup_list": nodup_list}


def get_context_by_one(smi_path): #输入一个路径，输出一个context列表，内容是文件里的每一行，感觉这个文件应该是sim_SimpleGenerator
    print(smi_path.split("/")[-1])
    if not os.path.exists(smi_path):
        return []
    with open(smi_path, "r") as f:
        context = f.readlines()
    print("single lines:", len(context))
    return context 


def run(smi_path, save_path, N_beam_search=10, if_full=False): #猜测smi_path是，save_path是score_SimpleGenerator
    if "{}" in smi_path:
        context = []
        files = glob.glob(smi_path.replace("{}", "*"))
        for i in files:
            context += get_context_by_one(i)
    else:
        context = get_context_by_one(smi_path)

    iter_N = N_beam_search + 3
    N_mol = int(len(context) / iter_N)
    if if_full and N_mol != 5007:
        return
    print(N_mol)
    start = time.time()
    with Pool() as pool:
        results = pool.map(
            worker,
            [
                (i, context, iter_N, N_beam_search)
                for i in tqdm(range(len(context) // iter_N))
            ],
        )

    target_list_all = []
    unstrict_list_all = []
    strict_list_all = []
    pred_list_all = []
    nodup_list_all = []

    if save_path is not None:
        with open(save_path.replace(save_path.split("/")[-1], "smiles_infer.txt"), 'w') as f:
            # json.dump(i, f)
            f.write(json.dumps(results, indent=4))

    for i in results:
        target = i["target"]
        unstrict_list = i["unstrict_list"]
        strict_list = i["strict_list"]
        pred_list = i["pred_list"]
        nodup_list = i["nodup_list"]

        target_list_all.append(target)
        unstrict_list_all.append(unstrict_list)
        strict_list_all.append(strict_list)
        pred_list_all.append(pred_list)
        nodup_list_all.append(nodup_list)

    if save_path is not None:
        f = open(save_path, "w")
    else:
        f = None
    print(time.time() - start)
    unstrict_list_all = np.array(unstrict_list_all)
    strict_list_all = np.array(strict_list_all)
    unstrict_list_all = unstrict_list_all.sum(0)
    strict_list_all = strict_list_all.sum(0)
    nodup_list_all = np.array(nodup_list_all)
    nodup_list_all = nodup_list_all.sum(0)
    print("total", N_mol)
    print("unstrict", unstrict_list_all)
    print("unstrict", unstrict_list_all / N_mol)
    print("strict", strict_list_all)
    print("strict", strict_list_all / N_mol)
    print("nodup_list_all", nodup_list_all)
    print("nodup_list_all", nodup_list_all / N_mol)
    print("\n")

    if f is not None:
        f.write("total " + str(N_mol))
        f.write("\nunstrict " + str(unstrict_list_all))
        f.write("\nunstrict " + str(unstrict_list_all / N_mol))
        f.write("\nstrict " + str(strict_list_all))
        f.write("\nstrict " + str(strict_list_all / N_mol))
        f.write("\nnodup_list_all " + str(nodup_list_all))
        f.write("\nnodup_list_all " + str(nodup_list_all / N_mol))
        f.write("\n")
    try:
        unstrict_list_tmp = [unstrict_list_all[0]]
        for i in range(1, len(unstrict_list_all)):
            unstrict_list_tmp.append(unstrict_list_tmp[i - 1] + unstrict_list_all[i])
        unstrict_list_tmp = np.array(unstrict_list_tmp)
    except:
        unstrict_list_tmp = 0
    strict_list_tmp = [strict_list_all[0]]
    for i in range(1, len(strict_list_all)):
        strict_list_tmp.append(strict_list_tmp[i - 1] + strict_list_all[i])
    strict_list_tmp = np.array(strict_list_tmp)

    nodup_list_tmp = [nodup_list_all[0]]
    for i in range(1, len(nodup_list_all)):
        nodup_list_tmp.append(nodup_list_tmp[i - 1] + nodup_list_all[i])
    nodup_list_tmp = np.array(nodup_list_tmp)

    print("unstrict", unstrict_list_tmp)
    print("unstrict", unstrict_list_tmp / N_mol)
    print("strict", strict_list_tmp)
    print("strict", strict_list_tmp / N_mol)
    print("nodup_list_all", nodup_list_tmp)
    print("nodup_list_all", nodup_list_tmp / N_mol)

    if f is not None:
        f.write("unstrict " + str(unstrict_list_tmp))
        f.write("\nunstrict " + str(unstrict_list_tmp / N_mol))
        f.write("\nstrict " + str(strict_list_tmp))
        f.write("\nstrict " + str(strict_list_tmp / N_mol))
        f.write("\nnodup_list_all " + str(nodup_list_tmp))
        f.write("\nnodup_list_all " + str(nodup_list_tmp / N_mol))
        f.close()


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     raise "ERROR"
    smi_path = '/public/home/zjh/dln/BioG2G/outputs_biochem_plus/BioG2G_unimolplus_uspto_50k_b16_2_l6_vnode1_wda_true_lp_0_0_nsum2_false_sep2_false_cls_false_hdegree_true_fofalse_sg_randomsmiles_lr_2.5e-4_wd_0.0_mp__20250521-235024/checkpoint_test/smi_SimpleGenerator_lp0.0_t1_20_bhs2_bss5_b1_biochem_plus1_0.txt'
    N_beam_search = 10
    # if len(sys.argv) >= 3:
    #     N_beam_search = int(sys.argv[2])
    # if "--if_full" in sys.argv:
    #     if_full = True
    # else:
    #     if_full = False
    # if len(sys.argv) >= 5:
    #     score_name = sys.argv[4]
    # else:
    #     score_name = "score"
    save_path = smi_path.replace(smi_path.split("/")[-1], "score")
    run(smi_path, save_path, N_beam_search=N_beam_search, if_full=False)
