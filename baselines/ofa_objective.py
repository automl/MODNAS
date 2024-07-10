from syne_tune import Reporter
import pickle
import torch
from predictors.help.loader import Data
from ofa.tutorial.accuracy_predictor import AccuracyPredictor
from predictors.help.net import  MetaLearner
report = Reporter()
import pickle
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
def reformat_ofa_arch(arch):
    arch_new = {}
    arch_new["d"] = []
    arch_new["e"] = []
    arch_new["ks"] = []
    for i in range(5):
        arch_new["d"].append(arch["depth"+str(i)])
        for j in range(4):
            arch_new["e"].append(arch["expand"+str(i)+str(j)])
            arch_new["ks"].append(arch["kernel_size"+str(i)+str(j)])
    arch_new["r"] = arch["resolution"]
    print(arch_new)
    return arch_new

def get_gt_stats_latency(device, help_loader):
    with open("/work/dlclarge1/sukthank-modnas/MODNAS_ICML/main/hat_supernet/MODNAS-patent/stats_ofa.pkl","rb") as f:
       #f = "/work/dlclarge1/sukthank-modnas/MODNAS_ICML/main/hat_supernet/MODNAS-patent/stats_ofa.pkl"
       stats = CPU_Unpickler(f).load()
    return stats[device]["max"].item(), stats[device]["min"].item()

def preprocess_for_predictor(predictor, arch_params):
        kernel_size_weights, expand_ratio_weights, depth_weights, resolution_weights = arch_params["ks"], arch_params["e"], arch_params["d"], arch_params["r"]
        depth_selected = []
        depth_list = [2,3,4]
        expand_list = [3,4,6]
        kernel_list = [3,5,7]
        #print(depth_weights)
        kernel_size_zeros = torch.zeros_like(kernel_size_weights).to(kernel_size_weights.device)
        expand_ratio_zeros = torch.zeros_like(expand_ratio_weights).to(kernel_size_weights.device)
        for i in range(5):
            argmax_depth = torch.argmax(depth_weights[i])
            depth_selected.append(depth_list[argmax_depth])
        start = 0
        end = max(depth_list)
        for i,d in enumerate(depth_selected):
            for j in range(start,start+d):
                expand_ratio_zeros[i,j,:] = expand_ratio_weights[i,j,:]
                kernel_size_zeros[i,j,:] = kernel_size_weights[i,j,:]
            for j in range(start+d, end):
                expand_ratio_zeros[i,j,:] = 0
                kernel_size_zeros[i,j,:] = 0
        resolution_weights = arch_params["r"]
        # set max
        #resolution_weights
        out = torch.cat([kernel_size_zeros.reshape(-1),expand_ratio_zeros.reshape(-1), resolution_weights.reshape(-1), depth_weights.reshape(-1)]).unsqueeze(0)
        return out
def convert_to_dict(arch_config):
    arch_param_depth = torch.zeros(5,3)
    arch_param_expand = torch.zeros(5,4, 3)
    arch_param_kernel = torch.zeros(5,4, 3)
    arch_param_resolution = torch.zeros(25)
    resolution_choices = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224]
    arch_param_resolution[resolution_choices.index(arch_config["r"])] = 1
    depth_list = [2,3,4]
    expand_list = [3,4,6]
    kernel_list = [3,5,7]
    start = 0
    for i in range(5):
        arch_param_depth[i,depth_list.index(arch_config["d"][i])] = 1
        for j in range(arch_config["d"][i]):
            arch_param_expand[i,j,expand_list.index(arch_config["e"][start])] = 1
            arch_param_kernel[i,j,kernel_list.index(arch_config["ks"][start])] = 1
            start+=1
        for j in range(arch_config["d"][i],4):
            start+=1
    return {"d":arch_param_depth, "e":arch_param_expand, "ks":arch_param_kernel, "r":arch_param_resolution}

def objective(arch, metric):
    arch = reformat_ofa_arch(arch)
    device = "cpu"
    
    acc_predictor = AccuracyPredictor(
    pretrained=True,
    device=device
    )
    help_loader = Data(mode="nas",data_path="/work/dlclarge1/sukthank-modnas/MODNAS_ICML/HELP/data/ofa",search_space="ofa", meta_valid_devices=[], meta_test_devices= [], meta_train_devices=["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","v100_1","v100_32", "v100_64", "titan_xp_1", "titan_xp_32", "titan_xp_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )
    latency_predictor = MetaLearner("ofa",True,10,100)
    latency_predictor.load_state_dict(torch.load("/work/dlclarge1/sukthank-modnas/MODNAS_ICML/MODNAS-patent/predictors/ofa/train/ofa_predictor_modified.pt",map_location="cpu"))
    hw_embed, _, _, _, _, _ = help_loader.get_task(metric)
    accuacy = acc_predictor.predict_accuracy([arch])
    #print(accuacy)
    error = 1-accuacy.item()
    normalized_error =  error
    arch_param = convert_to_dict(arch)
    arch_param = preprocess_for_predictor(latency_predictor, arch_param)
    latency = latency_predictor(arch_param, hw_embed)
    latency = latency.item()
    latency_max , latency_min = get_gt_stats_latency(metric, help_loader)
    normalized_latency = (latency-latency_min)/(latency_max-latency_min)
    report(error = normalized_error, latency = normalized_latency)

if __name__ == "__main__":
    import logging
    import argparse
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    block_len = 5
    max_depth = 4
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--st_checkpoint_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--metric", type=str, default="2080ti_1")
    parser.add_argument("--block_len", type=int, default= 5)
    parser.add_argument("--max_depth", type=int, default= 4)
    parser.add_argument("--resolution", type=int, default= 224)

    for i in range(block_len):
        parser.add_argument(f"--depth{i}", type=int, default=3)
        for j in range(max_depth):
            parser.add_argument(f"--kernel_size{i}{j}", type=int, default=5)
            parser.add_argument(f"--expand{i}{j}", type=int, default=3)
    args = parser.parse_args()
    args_dict = {}
    for i in range(block_len):
        args_dict[f"depth{i}"] = args.__dict__[f"depth{i}"]
        for j in range(max_depth):
            args_dict[f"kernel_size{i}{j}"] = args.__dict__[f"kernel_size{i}{j}"]
            args_dict[f"expand{i}{j}"] = args.__dict__[f"expand{i}{j}"]
    args_dict["resolution"] = args.resolution
    objective(args_dict, args.metric)
