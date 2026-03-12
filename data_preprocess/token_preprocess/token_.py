import torch
from transformers import BartTokenizer

# dict_path = "/mnt/vepfs/users/yaolin/BioG2G_/retro/USPTO50K_ALL/vocab.txt"
# tokenizer = BertWordPieceTokenizer(dict_path)
# max_seq_len = 512

# raw_str = "NC(=O)c1c[se]c([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)n1"
# raw_str = raw_str.replace('<unk>', '[UNK]')
# output = tokenizer.encode(raw_str)
# print(output)
# print(output.ids)

# c = tokenizer.decode(output.ids)
# print("**", c)
# print(output.type_ids)
# print(output.tokens)
# print(output.offsets)
# print(output.attention_mask)
# print(output.special_tokens_mask)
# print(output.overflowing)
def abcdd():
    path = "/mnt/vepfs/users/yaolin/BioG2G_/retro/USPTO50K_ALL/tokenizer-smiles-bart/vocab.json"
    import json
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
        load_dict = [i for i in load_dict.keys()]
        with open(path.replace("vocab.json","dict.txt"),"a+") as dump_f:
            for i in load_dict:
                dump_f.write(i)
                dump_f.write("\n")
        print(load_dict)
        load_dict = {load_dict[i]: i for i in range(len(load_dict))}
        print(load_dict)

        with open(path.replace(".json","1.json"),"w") as dump_f:
            json.dump(load_dict,dump_f)

# abcdd()
path = "/mnt/vepfs/users/yaolin/BioG2G_/retro/USPTO50K_ALL/tokenizer-smiles-bart/"
tokenizer = BartTokenizer.from_pretrained(path)
raw_str = "//\///NC(=O)c1c[SSese]c([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)n1"
raw_str = ""
print(raw_str)
output = tokenizer(raw_str)["input_ids"]
print(output)
# c = [tokenizer.decode(i) for i in output["input_ids"]]
# print(c)
# output = tokenizer.decode(output["input_ids"])
# print(output)


