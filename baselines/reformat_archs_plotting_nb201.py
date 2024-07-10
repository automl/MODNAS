experiment_str = "monb201"
methods=("RS" ,"Grid", "MOREA", "LS", "NSGA2", "LSBO", "RSBO", "MOASHA", "EHVI")
devices=("eyeriss","fpga")
seeds=(9001, 9002, 9003, 9004)
import pickle
import torch
import os
from search_spaces.nb201.model_search import NASBench201SearchSpace
def convert_arch_to_arch_param(arch):
    #print(arch)
    arch_param = torch.zeros([6, 5])
    for i in range(6):
        arch_param[i,arch["edge"+str(i)]] = 1
    return arch_param
restart = []
model = NASBench201SearchSpace(16, 5, 4, 10)#.cuda()
for method in methods:
    for device in devices:
        for seed in seeds:
            save_path = "nb201_synetune_baselines_3_objs/"+experiment_str+"_"+method+"_"+device+"_latency_"+str(device)+"_energy_"+str(seed)+"_archs.pkl"
            config_file = "results/monb2013obj_"+str(method)+"_"+str(device)+"_latency_"+str(device)+"_"+str(seed)+".pickle"
            if not os.path.isfile(config_file):
                restart.append(config_file)
                continue
            print("processing", device, seed, method)
            with open(config_file,"rb") as f:
                configs = pickle.load(f)
                configs = configs["configs"]
            archs  = []
            for c in configs:
                arch_param =convert_arch_to_arch_param(c)
                model.set_arch_params(arch_param)
                arch_str = model.genotype().tostr()
                archs.append(arch_str)
            with open(save_path,"wb") as f:
                pickle.dump(archs,f)
print(restart)