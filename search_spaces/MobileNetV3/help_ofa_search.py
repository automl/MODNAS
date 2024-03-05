from utils import run_ofa_search_help, circle_points
from optimizers.help.loader import Data
if __name__ == "__main__":
    rays = circle_points(24)
    devices = ["2080ti_1","2080ti_32","2080ti_64","titan_xp_32","titan_rtx_32","titan_xp_1","titan_rtx_1","titan_xp_64","v100_1","v100_32", "v100_64"] +  ["titan_rtx_64"]
    help_loader =  Data(mode="meta-train",data_path="datasets/help/ofa",search_space="ofa", meta_valid_devices=devices, meta_test_devices=devices, meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )
    devices = ["2080ti_32","2080ti_64","titan_xp_32","titan_rtx_32","titan_xp_1","titan_rtx_1","titan_xp_64","v100_1","v100_32", "v100_64"] +  ["titan_rtx_64"] + ["2080ti_1"]
    archs_help = {}
    import pickle
    for device in devices:
        print(f"Device: {device}")
        archs_help[device] = {}
        archs, accs, lats = run_ofa_search_help(rays, device, help_loader)
        archs_help[device]["archs"] = archs
        archs_help[device]["accs"] = accs
        archs_help[device]["lats"] = lats
        pickle.dump(archs_help, open("archs_help.pkl", "wb"))

