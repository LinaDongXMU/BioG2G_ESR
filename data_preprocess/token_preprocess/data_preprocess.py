import os
import json
from tqdm import tqdm
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 0 [PAD] 田
# 1 [CLS] 由
# 2 [SEP] 甲
# 3 [UNK] 申
# 253 [SEP2] 瘭

special_tokens = ["田", "由", "甲", "申", "瘭"]
# special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def get_dict(path, save_path, shift=30000):
    with open(path, "r") as f:
        context = f.readlines()
    with open(save_path, "w") as f_w:
        for idx, i in enumerate(context):
            f_w.write(str(idx) + " " + i.strip() + " " + chr(idx + shift) + "\n")


def tostring(tmp, shift=30000):
    tmp = tmp.strip().split(" ")
    tmp = [int(i) for i in tmp[1:-1]]
    return "".join([chr(i + shift) for i in tmp])


def toint(tmp, shift=30000):
    return [ord(i) - 30000 for i in tmp]


def tonewtxt(path, save_path):
    with open(save_path, "w") as f_w:
        for p in path:
            with open(p, "r") as f:
                context = f.readlines()
            for i in tqdm(context):
                str1 = tostring(i)
                f_w.write(str1 + "\n")


def to_after_sep2_txt(path, save_path):
    with open(save_path, "w") as f_w:
        for p in path:
            with open(p, "r") as f:
                context = f.readlines()
            for i in tqdm(context):
                str1 = i.split("瘭")
                if len(str1) == 2 and str1[1] != "":
                    f_w.write(str1[1])


def get_new_vocab(trans_txt, vocab_json):
    with open(vocab_json, "r") as f:
        dict_ = json.load(f)

    with open(trans_txt, "r") as f:
        trans = f.readlines()
        trans = [i.strip().split(" ") for i in trans]
        trans = {i[2]: i[1] for i in trans}
    new_dict = {}
    for k, v in dict_.items():
        k = " ".join([trans[i] for i in k])
        new_dict[k] = v

    with open(vocab_json.replace("vocab", "vocab_new"), "w") as json_file:
        json.dump(new_dict, json_file)

    with open(vocab_json.replace(".json", ".txt"), "w") as txt_file:
        txt_file.write("\n".join([i for i in new_dict.keys()]))


def fix_vocab_json(trans_txt, vocab_json):
    with open(vocab_json, "r") as f:
        dict_ = json.load(f)
        dict_ = list(dict_.keys())
        dict_ = [i for i in dict_ if len(i) > 1]

    with open(trans_txt, "r") as f:
        trans = f.readlines()
        trans = [i.strip().split(" ") for i in trans]
        trans = [i[2] for i in trans]

    total = trans + dict_
    total = {i: idx for idx, i in enumerate(total)}
    with open(
        vocab_json.replace("vocab", "vocab_fix"), "w", encoding="utf8"
    ) as json_file:
        json.dump(total, json_file, ensure_ascii=False)


def replace_sep2(path, word=" 253"):
    with open(path, "r") as f:
        text = f.read()
        new_text = text.replace(word, "")
    with open(path, "w") as f:
        f.write(new_text)


def build_token(files, save_folder, test_code=None):
    tokenizer = Tokenizer(BPE(unk_token="申"))
    if not isinstance(files, List):
        files = [files]
    trainer = BpeTrainer(
        vocab_size=10000, min_frequency=5000, special_tokens=special_tokens
    )
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(files, trainer)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    tokenizer.model.save(save_folder)
    if test_code is not None:
        encoding = tokenizer.encode(test_code)
        print("Encoded string: {}".format(encoding.tokens))
        print("Encoded ids: {}".format(encoding.ids))
        decoded = tokenizer.decode(encoding.ids)
        print("Decoded string: {}".format(decoded))

    fix_vocab_json(
        trans_txt,
        save_folder + "/vocab.json",
    )
    get_new_vocab(
        trans_txt,
        save_folder + "/vocab_fix.json",
    )

def init(tokenizer_path):
    tokenizer = Tokenizer(
        BPE.from_file(
            os.path.join(tokenizer_path, "vocab.json"),
            os.path.join(tokenizer_path, "merges.txt"),
        )
    )
    # tokenizer.enable_padding(
    #         pad_id=1, pad_token="<pad>", length=maxlen)
    # tokenizer.enable_truncation(max_length=maxlen)
    return tokenizer


def main(tokenizer_path):
    tokenizer = init(tokenizer_path)
    tmp = [
        "由男甶痈甶病甶疽痈甶疽痈甶痈畊病甶疽痊痈甶疽痈痍痈男痐痈甶痈甶疽痈甶疽痈甶疽痈男痈甶痈痌痈痒痈甶疾痘病甶病甸痆男疾甶病男痆甶疾病甶疾病画病痋病瘭甸疽痐病甲",
        "由甶甹病甶痊痈甶疽痈甶疽痈甶痈甶病甶疽症甶疽痋痈痍痈甶痏病男症甲",
        "由甶甶疽痈甶疽痈甶痈男疽病甶疽病甶疾病甶疾病男疽病甶疾甶疾病甶病甹病甹痊病甹痋病甶疾痏病痋病甶疽痕痈甶痈痘痈畊病甶痛病男症瘭甿痔病甲",
    ]
    a = tokenizer.encode(tmp[1])
    print(a.ids)
    print(a.tokens)
    b = tokenizer.decode(a.ids)
    print(b.replace(" ", "") == tmp[1])


def compare(a, b):
    with open(a, "r") as fa:
        dict_a = json.load(fa)
        dict_a = set(dict_a.keys())

    with open(b, "r") as fb:
        dict_b = json.load(fb)
        dict_b = set(dict_b.keys())

    intersection = dict_a.intersection(dict_b)
    only_set1 = dict_a - dict_b
    only_set2 = dict_b - dict_a
    print(len(intersection))
    print(len(only_set1))
    print(len(only_set2))


if __name__ == "__main__":
    pass
    # get_dict("/data/users/yaolin/token/dict_20230310.txt", "/data/users/yaolin/token/trans.txt")
    # replace_sep2("/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/50k_data_wosep2.txt")
    # compare(
    #     "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/tokenizer_230413_wosep/vocab_new.json",
    #     "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/tokenizer_230413_wsep/vocab_new.json",
    # )

    # path = [
    #     # "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/50k_data_wsep2.txt",
    #     "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/50k_data_wosep2.txt",
    # ]
    save_path = "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/50k_data_save_aftersep2.txt"
    save_folder = "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/tokenizers/tokenizer_50k_230413_aftsep2"
    trans_txt = "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/trans.txt"

    # tonewtxt(path, save_path)

    # to_after_sep2_txt(
    #     [
    #         "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/50k_data_save_wsep.txt"
    #     ],
    #     "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/50k_data_save_aftersep2.txt",
    # )

    build_token(save_path, save_folder)
    # main(
    #     "/data/users/yaolin/BioG2G/BioG2G_model/data_preprocess/token_preprocess/data/tokenizer_230413_wosep"
    # )

