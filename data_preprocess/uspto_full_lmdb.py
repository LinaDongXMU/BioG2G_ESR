import csv
import pandas as pd
from basic import brief_dataset, csv_file_read

brief_dataset("/data/users/yaolin/BioG2G_/retro/USPTO50K_brief_20230227/train.lmdb")
# brief_dataset("/data/projects/BioG2G/dataset/synthesis/uspto_full_lmdb/uspto_testT.lmdb")

# a = csv_file_read("/data/projects/BioG2G/dataset/synthesis/uspto-full/extract/uspto_valT.csv")
# for i in a[:3]:
#     print(i)
# def gen43D(path, csv_path, len_=None):
#     with open(csv_path, "w") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["smi"])

#         with open(path, "r", encoding="utf-8") as csvfile:
#             reader = csv.DictReader(csvfile)
#             for row in tqdm(reader):
#                 product, reactant = row["input"], row["target"]
#                 if check_smiles(product, reactant, len_=len_):
#                     writer.writerow([0, 0, reactant + ">>" + product])
