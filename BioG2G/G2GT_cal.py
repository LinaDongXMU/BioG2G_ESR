from utils.G2G_cal_test import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise "ERROR"
    if ".txt" in sys.argv[1]:
        smi_path = sys.argv[1]
        N_beam_search = 10
        if len(sys.argv) >= 3:
            N_beam_search = int(sys.argv[2])
        if "--if_full" in sys.argv:
            if_full = True
        else:
            if_full = False
        if '--score-name' in sys.argv:
            idx = sys.argv.index('--score-name') + 1
            score_name = sys.argv[idx]
            save_path = smi_path.replace(smi_path.split("/")[-1], score_name)
        else:
            # score_name = "score"
            score_name = smi_path.split("/")[-1].replace("smi", "score")
            save_path = smi_path.replace(smi_path.split("/")[-1], score_name)
        # save_path = None
        # smi_path: /data/users/qifanyu/nag2g_BioG2G/BioG2G/outputs/weights/synthesis/2023-06-30/BioG2G_G2GT_uspto_50k_b16_1_l6_vnode1_wda_true_lp_0_0_nsum2_false_sep2_false_cls_false_hdegree_true_sg_randomsmiles_lr_2.5e-4_wd_0.0_mp__20230630-134327/smi_SequenceScoreGeneratorBeamSearch_lp0.0_t1_10_bhs2_bss5_b128_USPTO50K_raw_20230220_{}.txt
        # save_path: /data/users/qifanyu/nag2g_BioG2G/BioG2G/outputs/weights/synthesis/2023-06-30/BioG2G_G2GT_uspto_50k_b16_1_l6_vnode1_wda_true_lp_0_0_nsum2_false_sep2_false_cls_false_hdegree_true_sg_randomsmiles_lr_2.5e-4_wd_0.0_mp__20230630-134327/score_SequenceScoreGeneratorBeamSearch_lp0.0_t1_10_bhs2_bss5_b128_USPTO50K_raw_20230220_{}.txt
        # print("smi_path:", smi_path)
        # print("save_path:", save_path)
        # exit()

        if "--S" in sys.argv:
            task = 'synthesis'
        else:
            task = 'retrosynthesis'
        run(smi_path, save_path, N_beam_search=N_beam_search,
            if_full=if_full, task=task)
        if "--failed_samples" in sys.argv:
            write_failed_samples(save_path)
            exit()
    else:
        path = sys.argv[1]
        # txt_name = "smi_lp0.0_t0_10_b256.txt"
        dirs = [
            os.path.join(path, i)
            for i in os.listdir(path)
            if "checkpoint_" in i and ".pt" not in i
        ]
        for i in dirs:
            dirs2 = [os.path.join(i, j) for j in os.listdir(
                i) if ".txt" in j and "smi" in j]
            for j in dirs2:
                orders = "python G2GT_cal.py {} &".format(j)
                print(orders)
                os.system(orders)
