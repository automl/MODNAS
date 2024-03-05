import torch
from utils import circle_points
from optimizers.help.loader import Data

def arch_param_to_config(arch_param):
    config = {}
    depth_choices = [2,3,4]
    ks_choices = [3,5,7]
    e_choices = [3,4,6]
    imlist = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224]
    config["d"] = []
    config["e"] = []
    config["ks"] = []
    config["r"] = imlist[torch.argmax(arch_param["r"])]
    for i in range(5):
        config["d"].append(depth_choices[torch.argmax(arch_param["d"][i])])
    for i in range(5):
        for j in range(config["d"][i]):
            config["e"].append(e_choices[torch.argmax(arch_param["e"][i][j])])
            config["ks"].append(ks_choices[torch.argmax(arch_param["ks"][i][j])])
    return config
checkpoint = torch.load("path/to/supernet", map_location="cpu")
hpn = checkpoint["state_dict_hpn"]
from hypernetworks.hpns_ofa import HyperNetworkwores, MetaHyperNetwork, convert_to_dict, HyperNetwork
hpn_model = MetaHyperNetwork(5,4,3,3,3,25,50,10,HyperNetwork)
hpn_model.load_state_dict(hpn)

from search.utils import circle_points
scalarizations = circle_points(24)
help_loader = Data(mode="meta-train",data_path="datasets/help/ofa",search_space="ofa", meta_valid_devices=[], meta_test_devices= [], meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )

archs_ours = {}

devices = help_loader.meta_train_devices
for device in devices:
    hw_embed, _, _, _, _, _ = help_loader.get_task(device)
    hw_embed = torch.squeeze(torch.tensor(hw_embed)).unsqueeze(0)
    archs_ours[device] = {}
    for s in scalarizations:
        arch_param = hpn_model(torch.tensor(s).unsqueeze(0), hw_embed)
        arch_param = convert_to_dict(arch_param)
        config = arch_param_to_config(arch_param)
        print(config)
        archs_ours[device][str(s)] = config
import pickle
pickle.dump(archs_ours, open("archs_ofa.pkl", "wb"))