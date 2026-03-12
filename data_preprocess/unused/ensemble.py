import torch
import collections


def integration(model_list, path_name):
    worker_state_dict = [torch.load(x, map_location="cpu")["model"] for x in model_list]
    print(worker_state_dict[0].keys())
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        print("key is {}".format(key))
        key_sum = 0
        for i in range(len(model_list)):
            key_sum += worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / float(len(model_list))

    tmp = torch.load(model_list[0], map_location="cpu")
    tmp["model"] = fed_state_dict
    torch.save(tmp, path_name)

path_name="/data/users/yaolin/BioG2G/outputs/weights_20230328/BioG2G_G2GT_uspto_50k_b16_1_l6_vnode1_wda_false_lp_0_2_nsum2_true_sep2_true_cls_false_hdegree_true_sg_randomsmiles_lr_2.5e-4_wd_0.0_mp__20230407-033209/16_360000.pt"
model_list = [
    "/data/users/yaolin/BioG2G/outputs/weights_20230328/BioG2G_G2GT_uspto_50k_b16_1_l6_vnode1_wda_false_lp_0_2_nsum2_true_sep2_true_cls_false_hdegree_true_sg_randomsmiles_lr_2.5e-4_wd_0.0_mp__20230407-033209/checkpoint_64_160000.pt",
    "/data/users/yaolin/BioG2G/outputs/weights_20230328/BioG2G_G2GT_uspto_50k_b16_1_l6_vnode1_wda_false_lp_0_2_nsum2_true_sep2_true_cls_false_hdegree_true_sg_randomsmiles_lr_2.5e-4_wd_0.0_mp__20230407-033209/checkpoint_144_360000.pt",
]
integration(model_list, path_name)