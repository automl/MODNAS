#!/bin/bash
methods=("MOREA", "LS", "NSGA2", "LSBO", "RSBO", "MOASHA","EHVI")
devices=["cpu_xeon"]
restart = []
import os 
import pickle
import numpy as np
import yaml
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

def reformat_for_yaml(arch):
    arch_dict = {}
    arch_dict["encoder-embed-dim-subtransformer"] = arch["encoder"]["encoder_embed_dim"]
    arch_dict["decoder-embed-dim-subtransformer"] = arch["decoder"]["decoder_embed_dim"]
    arch_dict["encoder-layer-num-subtransformer"] = arch["encoder"]["encoder_layer_num"]
    arch_dict["decoder-layer-num-subtransformer"] = arch["decoder"]["decoder_layer_num"]
    arch_dict["encoder-ffn-embed-dim-all-subtransformer"] = arch["encoder"]["encoder_ffn_embed_dim"]
    arch_dict["encoder-self-attention-heads-all-subtransformer"] = arch["encoder"]["encoder_self_attention_heads"]
    arch_dict["decoder-ffn-embed-dim-all-subtransformer"] = arch["decoder"]["decoder_ffn_embed_dim"]
    arch_dict["decoder-self-attention-heads-all-subtransformer"] = arch["decoder"]["decoder_self_attention_heads"]
    arch_dict["decoder-ende-attention-heads-all-subtransformer"] = arch["decoder"]["decoder_ende_attention_heads"]
    arch_dict["decoder-arbitrary-ende-attn-all-subtransformer"] = arch["decoder"]["decoder_arbitrary_ende_attn"]
    return arch_dict
def reformat_hat_arch(arch):
    arch_dict = {}
    arch_dict['encoder'] = {}
    arch_dict['decoder'] = {}
    arch_dict['encoder']['encoder_embed_dim'] = int(arch["encoder-embed-choice_"])
    arch_dict['decoder']['decoder_embed_dim'] = int(arch["decoder-embed-choice_"])
    arch_dict['encoder']['encoder_layer_num'] = int(arch["encoder-layer-num-choice_"])
    arch_dict['decoder']['decoder_layer_num'] = int(arch["decoder-layer-num-choice_"])
    arch_dict['encoder'][f'encoder_ffn_embed_dim'] = []
    arch_dict['encoder'][f'encoder_self_attention_heads'] = []
    arch_dict['decoder'][f'decoder_ffn_embed_dim'] = []
    arch_dict['decoder'][f'decoder_self_attention_heads'] = []
    arch_dict['decoder'][f'decoder_ende_attention_heads'] = []
    arch_dict['decoder'][f'decoder_arbitrary_ende_attn'] = []
    for i in range(arch_dict['decoder']['decoder_layer_num']):
        arch_dict['decoder'][f'decoder_ffn_embed_dim'].append(int(arch[f'decoder-ffn-embed-choice_{i}']))
        arch_dict['decoder'][f'decoder_self_attention_heads'].append(int(arch[f'decoder-self-attention-heads-choice_{i}']))
        arch_dict['decoder'][f'decoder_ende_attention_heads'].append(int(arch[f'decoder-ende-attention-heads-choice_{i}']))
        arch_dict['decoder'][f'decoder_arbitrary_ende_attn'].append(int(arch[f'decoder-arbitrary-ende-attn-choice_{i}']))
    for i in range(arch_dict['encoder']['encoder_layer_num']):
        arch_dict['encoder'][f'encoder_ffn_embed_dim'].append(int(arch[f'encoder-ffn-embed-choice_{i}']))
        arch_dict['encoder'][f'encoder_self_attention_heads'].append(int(arch[f'encoder-self-attention-heads-choice_{i}']))
    return arch_dict
experiment_str = "mohat"
error_dict = {}
def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val
for method in methods:
    for device in devices:
            save_path = "hat_synetune_baselines/"+experiment_str+"_"+method+"_"+device+"_archs.pkl"
            config_file = "results2/mohat_"+str(method)+"_"+str(device)+".pickle"
            print(config_file)
            if not os.path.isfile(config_file):
                restart.append(config_file)
                continue
            print("processing", device, method)
            with open(config_file,"rb") as f:
                configs = pickle.load(f)
            print(method,device,len(configs["configs"]))
            #print(configs.keys())
            #print(configs["configs"][0])
            lat = configs["memory"]
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
                    configs_pareto.append(reformat_hat_arch(config)) 
            print(len(configs_pareto))
            lat_pareto = [lat[i] for i in range(len(lat)) if pareto[i] == True]
            # write each arch to yaml file
            #print(lat_pareto)
            with open(save_path,"wb") as f:
                pickle.dump(configs_pareto,f)
            # make baseline dir
            if not os.path.exists("hat_baseline_archs"):
                os.makedirs("hat_baseline_archs")
            if not os.path.exists("hat_baseline_archs"+"/"+str(method)):
                os.makedirs("hat_baseline_archs"+"/"+str(method))
            for i in range(len(configs_pareto)):
                yaml_path = "hat_baseline_archs"+"/"+str(method)+"/"+str(device)+"_"+str(i)+".yaml"
                config_yaml = reformat_for_yaml(configs_pareto[i])
                with open(yaml_path,"w") as f:
                    yaml.dump(config_yaml,f, default_flow_style=None)
                with open("hat_baseline_archs"+"/"+str(method)+"/"+str(device)+"_"+str(i)+"_latency.pkl","wb") as f:
                 #if method == "LS" or method == "EHVI":
                 #   lat_denorm = denormalize(lat_pareto[i],2000,10000)
                 ##else:
                 lat_denorm = lat_pareto[i]  
                 print(lat_denorm)
                 pickle.dump(lat_denorm,f)
print(restart)
            

