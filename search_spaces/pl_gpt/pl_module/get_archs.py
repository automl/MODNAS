from predictors.gpt.hw_loader import search_spaces, HWDataset

def get_arch_config(arch_params,choices):
        layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param = arch_params

        argmax_layer = torch.argmax(layer_param, dim=-1)
        layer_selected = layer_param[argmax_layer]
        layer_num = choices['n_layer_choices'][argmax_layer]
        argmax_embed_dim = torch.argmax(embed_dim_param, dim=-1)
        embed_dim_selected = embed_dim_param[argmax_embed_dim]
        #print(argmax_embed_dim)
        embed_dim_selected = choices['embed_dim_choices'][argmax_embed_dim]
        argmax_bias = torch.argmax(bias_param, dim=-1)
        bias_selected = bias_param[argmax_bias]
        bias_selected = choices['bias_choices'][argmax_bias]
        argmax_head = torch.argmax(head_param, dim=-1)
        heads_selected = head_param[:,argmax_head]
        heads_selected = [choices['n_head_choices'][i] for i in argmax_head]
        argmax_mlp_ratio = torch.argmax(mlp_ratio_param, dim=-1)
        mlp_ratio_selected = mlp_ratio_param[:,argmax_mlp_ratio]
        mlp_ratio_selected = [choices['mlp_ratio_choices'][i] for i in argmax_mlp_ratio]
        intermediate_size = [embed_dim_selected * mlp_ratio for mlp_ratio in mlp_ratio_selected]

        sampled_arch = {"sample_embed_dim": embed_dim_selected, "sample_n_layer": layer_num, "sample_n_head": heads_selected, 'sample_mlp_ratio':mlp_ratio_selected, "sample_bias":bias_selected}
        return sampled_arch

from hypernetworks.hpn_gpt import MetaHyperNetwork
import torch
hpn = MetaHyperNetwork(search_spaces["s"])
#state_dict = 
#state_dict = "/p/scratch/ccstdl/sukthanker1/MODNAS-patent/experiments/owt_small_modnas_init_hpn_lower_lr_fixnorm/owt_small_modnas_init_hpn_lower_lr_fixnorm/default_juls/last.ckpt"
#state_dict = "/p/scratch/ccstdl/sukthanker1/MODNAS-patent/experiments/owt_small_modnas/owt_small_modnas/default_juls/checkpoint-03-0.ckpt"
state_dict = "/p/scratch/ccstdl/sukthanker1/MODNAS-patent/experiments/owt_small_modnas_test/owt_small_modnas_test/default_juls/last.ckpt"
state_dict_hpn = torch.load(state_dict)
# filter out hpn
print(list(state_dict_hpn.keys()))
hpn_keys = {}
for k in state_dict_hpn['state_dict'].keys():
    if k.startswith("hpn."):
        hpn_keys[k[4:]] = state_dict_hpn['state_dict'][k]
hpn.load_state_dict(hpn_keys)
from search.utils import circle_points
hwdset = HWDataset()
scalarizations = circle_points(24)

hpn = hpn.cuda()
all_archs = {}
devices_all = hwdset.gpus
for device in devices_all:
 batch_hw = hwdset.sample_batch(device, 1, "train")[2].float().cuda()
 all_archs[device]=[]
 count = {}
 count["384"] = 0
 count["768"] = 0
 count["192"] = 0
 for s in scalarizations:
    scal = torch.tensor(s).unsqueeze(0).cuda()
    arch_params = hpn(scal,batch_hw)
    arch = get_arch_config(arch_params, search_spaces['s'])
    if arch not in all_archs[device]:
       all_archs[device].append(arch)
       if arch["sample_embed_dim"] == 384: #and arch["sample_n_layer"] == 12:
           count["384"]+=1
       elif arch["sample_embed_dim"] == 768: #and arch["sample_n_layer"] == 12:
           count["768"]+=1
       elif arch["sample_embed_dim"] == 192: # and arch["sample_n_layer"] == 12:
           count["192"]+=1
 print(count)
import pickle
with open("archs_modnas_init_hpn_test_3.pkl","wb") as f:
    #print(all_archs)
    pickle.dump(all_archs,f)
