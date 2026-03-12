import json
import csv
from tqdm import tqdm

def json2csv(json_path, outfile):
    list_product = []
    if not isinstance(json_path, list):
        json_path = [json_path]

    for p in json_path:
        with open(p) as f:
            lines = f.readlines()
        for i in lines:
            result=json.loads(i)
            smiles = result["ori_smiles"].split('>')
            list_product.append(smiles[-1])

    list_2 = list(set(list_product))
    header = ["smi"]
    with open(outfile, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in tqdm(list_2):
            writer.writerow([i])

# def csv2csv(path, outfile):
#     list_ = []
#     for p in path:
#         row_list = csv_file_read(p)
#         for i in row_list:
#             list_ += get_from_raw_string(i)
#     list_ = list(set(list_))
#     header = ["smi"]
#     with open(outfile, "w", encoding="UTF8") as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         for i in tqdm(list_):
#             writer.writerow([i])



def lmdb2csv_bek(path, outfile, name="smiles_target"):
    list_2 = []
    if not isinstance(path, list):
        path = [path]

    for p in path:
        dataset = LMDBDataset(p)
        print(len(dataset))

        for i in tqdm(range(len(dataset))):
            if name == "smiles_train" or name == "all":
                reactant = "".join(dataset[i]["smiles_train"]).split(".")
                for k in reactant:
                    list_2.append(k)
            if name == "smiles_target" or name == "all":
                target = "".join(dataset[i]["smiles_target"]).split(".")[0]
                list_2.append(target)

    list_2 = list(set(list_2))

    header = ["smi"]
    with open(outfile, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in tqdm(list_2):
            writer.writerow([i])


def lmdb2json(path, outfile):
    if not isinstance(path, list):
        path = [path]
    list_ = []
    for p in path:
        dataset = LMDBDataset(p)
        print(len(dataset))

        for i in tqdm(range(len(dataset))):
            tmp = dataset[i]
            tmp.pop("target_coordinates")
            tmp.pop("smiles_train")
            tmp.pop("smiles_target")
            tmp.pop("smiles_target_label")
            tmp.pop("target_atoms")

            list_.append(tmp)

    with open(outfile, "w") as f:
        f.write(json.dumps(list_))
        f.close()


def lmdb2token(path, name="all"):
    path_d = "/mnt/vepfs/users/yaolin/BioG2G_/retro/USPTO50K_ALL/tokenizer-smiles-bart"
    tokenizer = BartTokenizer.from_pretrained(path_d)
    list_2 = []
    if not isinstance(path, list):
        path = [path]

    for p in path:
        dataset = LMDBDataset(p)
        print(len(dataset))

        for i in tqdm(range(len(dataset))):
            if name == "smiles_train" or name == "all":
                reactant = "".join(dataset[i]["smiles_train"]).split(".")
                for k in reactant:
                    list_2.append(k)
            if name == "smiles_target" or name == "all":
                target = "".join(dataset[i]["smiles_target"]).split(".")[0]
                list_2.append(target)

    list_2 = list(set(list_2))
    # for k in list_2:
    #     if "Se" in k:
    #         print(k)

    token_list = []
    for i in tqdm(list_2):
        token_list += smi_tokenizer(i, tokenizer)

    token_list = list(set(token_list))
    print(token_list, len(token_list))
    return set(token_list)


# json_path = "/data/users/yaolin/BioG2G_/pistachio_simplify.json"
# outfile = "/data/users/yaolin/BioG2G_/pistachio_simplify.csv"
# json2csv(json_path, outfile)