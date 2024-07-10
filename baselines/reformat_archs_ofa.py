#!/bin/bash
methods=("RS","Grid","MOREA","LS","NSGA2","LSBO","RSBO","MOASHA","EHVI")
devices=("2080ti_1","2080ti_32","2080ti_64","titan_xp_32","titan_rtx_32","titan_xp_1","titan_rtx_1","titan_xp_64","v100_1","v100_32","v100_64","titan_rtx_64")
restart = []
import os 
import pickle
experiment_str = "moofa"
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

import numpy as np 
def get_pareto_optimal(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto

for method in methods:
    for device in devices:
            save_path = "ofa_synetune_baselines/"+experiment_str+"_"+method+"_"+device+"_archs.pkl"
            config_file = "results2/moofa_"+str(method)+"_"+str(device)+".pickle"
            print(config_file)
            if not os.path.isfile(config_file):
                restart.append(config_file)
                continue
            print("processing", device, method)
            with open(config_file,"rb") as f:
                configs = pickle.load(f)
            lat = configs["latency"]
            err = configs["error"]
            err_argmin = [e.argmin() for e in err]
            lat = [lat[i][idx] for i, idx in enumerate(err_argmin)]
            err = [err[i][idx] for i, idx in enumerate(err_argmin)]   
            err = np.array(err).reshape(len(err),1)
            lat = np.array(lat).reshape(len(err),1)
            err_lat = np.concatenate([err,lat],axis=-1)   
            pareto = get_pareto_optimal(err_lat)
            configs_pareto = []
            for i,config in enumerate(configs["configs"]):
                if pareto[i] == True:
                    configs_pareto.append(reformat_ofa_arch(config)) 
            print(len(configs_pareto))
            with open (save_path,"wb") as f:
                pickle.dump(configs_pareto,f)
print(restart)
            

