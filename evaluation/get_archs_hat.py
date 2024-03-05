from  search_spaces.hat.fairseq import utils
import argparse, yaml
from search_spaces.hat.fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
import pickle
import torch
def get_hw_embed_dict(task):
    pickle_path = "hypernetwork_data_utils/hat/hat_lats_vector_dict_"+ task +".pkl"
    with open(pickle_path, 'rb') as f:
        hw_embed_dict = pickle.load(f)
    return hw_embed_dict

def preprocess_for_predictor(arch_param, choices):

        arch_param_encoder_dim = arch_param["encoder-embed-dim"]
        arch_param_decoder_dim = arch_param["decoder-embed-dim"]
        arch_param_encoder_layer_num = arch_param["encoder-layer-num"]
        arch_param_decoder_layer_num = arch_param["decoder-layer-num"]
        arch_param_encoder_ffn_embed_dim = torch.zeros_like(arch_param["encoder-ffn-embed-dim"]).cuda()
        layer_num_encoder = choices["encoder-layer-num-choice"][torch.argmax(arch_param_encoder_layer_num).item()]
        for i in range(layer_num_encoder):
             arch_param_encoder_ffn_embed_dim[i]= arch_param["encoder-ffn-embed-dim"][i]
        arch_param_decoder_ffn_embed_dim = torch.zeros_like(arch_param["decoder-ffn-embed-dim"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_ffn_embed_dim[i]= arch_param["decoder-ffn-embed-dim"][i]
        arch_param_encoder_self_attention_heads = torch.zeros_like(arch_param["encoder-self-attention-heads"]).cuda()
        layer_num_encoder = choices["encoder-layer-num-choice"][torch.argmax(arch_param_encoder_layer_num).item()]
        for i in range(layer_num_encoder):
                arch_param_encoder_self_attention_heads[i]= arch_param["encoder-self-attention-heads"][i]
        arch_param_decoder_self_attention_heads = torch.zeros_like(arch_param["decoder-self-attention-heads"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_self_attention_heads[i]= arch_param["decoder-self-attention-heads"][i]
        arch_param_decoder_ende_attention_heads = torch.zeros_like(arch_param["decoder-ende-attention-heads"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_ende_attention_heads[i]= arch_param["decoder-ende-attention-heads"][i]
        arch_param_decoder_arbitrary_ende_attn = torch.zeros_like(arch_param["decoder-arbitrary-ende-attn"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_arbitrary_ende_attn[i]= arch_param["decoder-arbitrary-ende-attn"][i]
        


        #arch_param_decoder_ffn_embed_dim = arch_param["decoder-ffn-embed-dim"]
        #arch_param_encoder_self_attention_heads = arch_param["encoder-self-attention-heads"]
        #arch_param_decoder_self_attention_heads = arch_param["decoder-self-attention-heads"]
        #arch_param_decoder_ende_attention_heads = arch_param["decoder-ende-attention-heads"]
        #arch_param_decoder_arbitrary_ende_attn = arch_param["decoder-arbitrary-ende-attn"]
        # concatenate all in flat list
        arch_param_list = []
        arch_param_list.append(arch_param_encoder_dim)
        arch_param_list.append(arch_param_decoder_dim)
        arch_param_list.append(arch_param_encoder_layer_num)
        arch_param_list.append(arch_param_decoder_layer_num)
        arch_param_list.append(arch_param_encoder_ffn_embed_dim.flatten())
        arch_param_list.append(arch_param_decoder_ffn_embed_dim.flatten())
        arch_param_list.append(arch_param_encoder_self_attention_heads.flatten())
        arch_param_list.append(arch_param_decoder_self_attention_heads.flatten())
        arch_param_list.append(arch_param_decoder_ende_attention_heads.flatten())
        arch_param_list.append(arch_param_decoder_arbitrary_ende_attn.flatten())
        # concat all in one tensor
        arch_param_tensor = torch.cat(arch_param_list, dim=-1).to(arch_param_encoder_dim.device)
        #print(arch_param_tensor)
        return arch_param_tensor
    
def arch_param_to_config(arch_param, space):
    config = {}
    config["encoder-embed-dim"] = space["encoder-embed-choice"][torch.argmax(arch_param["encoder-embed-dim"])]
    config["decoder-embed-dim"] = space["decoder-embed-choice"][torch.argmax(arch_param["decoder-embed-dim"])]
    config["encoder-layer-num"] = space["encoder-layer-num-choice"][torch.argmax(arch_param["encoder-layer-num"])]
    config["decoder-layer-num"] = space["decoder-layer-num-choice"][torch.argmax(arch_param["decoder-layer-num"])]
    config["encoder-ffn-embed-dim"] = [space["encoder-ffn-embed-dim-choice"][i] for i in torch.argmax(arch_param["encoder-ffn-embed-dim"], dim=-1)]
    config["decoder-ffn-embed-dim"] = [space["decoder-ffn-embed-dim-choice"][i] for i in torch.argmax(arch_param["decoder-ffn-embed-dim"], dim=-1)]
    config["encoder-self-attention-heads"] = [space["encoder-self-attention-heads-choice"][i] for i in torch.argmax(arch_param["encoder-self-attention-heads"], dim=-1)]
    config["decoder-self-attention-heads"] = [space["decoder-self-attention-heads-choice"][i] for i in torch.argmax(arch_param["decoder-self-attention-heads"], dim=-1)]
    config["decoder-ende-attention-heads"] = [space["decoder-ende-attention-heads-choice"][i] for i in torch.argmax(arch_param["decoder-ende-attention-heads"], dim=-1)]
    config["decoder-arbitrary-ende-attn"] = [space["decoder-arbitrary-ende-attn-choice"][i] for i in torch.argmax(arch_param["decoder-arbitrary-ende-attn"], dim=-1)]
    return config

def discretize_arch_param(arch_param):
    arch_param_discretized = {}
    arch_param_discretized["encoder-embed-dim"] = torch.zeros_like(arch_param["encoder-embed-dim"])
    arch_param_discretized["decoder-embed-dim"] = torch.zeros_like(arch_param["decoder-embed-dim"])
    arch_param_discretized["encoder-layer-num"] = torch.zeros_like(arch_param["encoder-layer-num"])
    arch_param_discretized["decoder-layer-num"] = torch.zeros_like(arch_param["decoder-layer-num"])
    arch_param_discretized["encoder-ffn-embed-dim"] = torch.zeros_like(arch_param["encoder-ffn-embed-dim"])
    arch_param_discretized["decoder-ffn-embed-dim"] = torch.zeros_like(arch_param["decoder-ffn-embed-dim"])
    arch_param_discretized["encoder-self-attention-heads"] = torch.zeros_like(arch_param["encoder-self-attention-heads"])
    arch_param_discretized["decoder-self-attention-heads"] = torch.zeros_like(arch_param["decoder-self-attention-heads"])
    arch_param_discretized["decoder-ende-attention-heads"] = torch.zeros_like(arch_param["decoder-ende-attention-heads"])
    arch_param_discretized["decoder-arbitrary-ende-attn"] = torch.zeros_like(arch_param["decoder-arbitrary-ende-attn"])
    arch_param_discretized["encoder-embed-dim"][torch.argmax(arch_param["encoder-embed-dim"])] = 1
    arch_param_discretized["decoder-embed-dim"][torch.argmax(arch_param["decoder-embed-dim"])] = 1
    arch_param_discretized["encoder-layer-num"][torch.argmax(arch_param["encoder-layer-num"])] = 1
    arch_param_discretized["decoder-layer-num"][torch.argmax(arch_param["decoder-layer-num"])] = 1
    argmax = torch.argmax(arch_param["encoder-ffn-embed-dim"], dim=-1)
    arch_param_discretized["encoder-ffn-embed-dim"][torch.arange(arch_param["encoder-ffn-embed-dim"].shape[0]), argmax] = 1
    argmax = torch.argmax(arch_param["decoder-ffn-embed-dim"], dim=-1)
    arch_param_discretized["decoder-ffn-embed-dim"][torch.arange(arch_param["decoder-ffn-embed-dim"].shape[0]), argmax] = 1
    argmax = torch.argmax(arch_param["encoder-self-attention-heads"], dim=-1)
    arch_param_discretized["encoder-self-attention-heads"][torch.arange(arch_param["encoder-self-attention-heads"].shape[0]), argmax] = 1
    argmax = torch.argmax(arch_param["decoder-self-attention-heads"], dim=-1)
    arch_param_discretized["decoder-self-attention-heads"][torch.arange(arch_param["decoder-self-attention-heads"].shape[0]), argmax] = 1
    argmax = torch.argmax(arch_param["decoder-ende-attention-heads"], dim=-1)
    arch_param_discretized["decoder-ende-attention-heads"][torch.arange(arch_param["decoder-ende-attention-heads"].shape[0]), argmax] = 1
    argmax = torch.argmax(arch_param["decoder-arbitrary-ende-attn"], dim=-1)
    arch_param_discretized["decoder-arbitrary-ende-attn"][torch.arange(arch_param["decoder-arbitrary-ende-attn"].shape[0]), argmax] = 1
    return arch_param_discretized

parser = options.get_training_parser()
parser.add_argument('--config-file')
options.add_generation_args(parser)
args = parser.parse_args()
# add test that len(args.names)==len(args.ages) ??
with open(args.config_file) as f:
    ydict = yaml.load(f)
    # {'county': 'somewhere', 'names': ['Bob', 'Jill'], 'ages': [22, 31]}
    # add list attributes from args to the corresponding ydict values
for k,v in ydict.items():
    av = getattr(args,k,None)
    if av and isinstance(v, list):
          v.extend(av)
search_space = utils.get_space("space0")
print(args)

from hypernetworks.models.hpn_hat import MetaHyperNetwork, convert_arch_params_to_dict
from search_spaces.hat.fairseq.tasks.translation import TranslationTask
from  search_spaces.hat.fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
   
#args.arch = "transformersuper_iwslt_de_en_2"
#args.task = "translation"
args.left_pad_source = "False"
args.left_pad_target = "False"
args.data = "data/binary/wmt14_en_de/"
args.source_lang = "en"
args.target_lang = "fr"
args.pred_save = "wmt14.en-de"
args.qkv_dim = 512
args.encoder_layers = 6
args.decoder_layers = 6
args.encoder_attention_heads = 4
args.decoder_attention_heads = 4

args.vocab_original_scaling = 1.0
args.get_attn = True
print(args.arch)
task = tasks.setup_task(args)
model = task.build_model(args).half()
from predictors.hat.train.latency_predictor import Net
metahpn = MetaHyperNetwork(search_space, num_random_devices=50)#.half()
metahpn.cuda()
metahpn.eval()
import torch
task = "wmt14_en_de"
state = torch.load("path/to/checkpoint.pth", map_location="cpu")["hypernetwork"]
metahpn.load_state_dict(state)
predictor = Net(101, 400, 6, hw_embed_on=True, hw_embed_dim=10).cuda()
state_predictor = torch.load("predictor_data_utils/hat/"+task+"_one_hot_"+".pt")
predictor.load_state_dict(state_predictor)
from hypernetworks.models.hpn_hat import convert_arch_params_to_dict
from utils import circle_points
scalarizations = circle_points(24)
devices = ["cpu_xeon", "gpu_titanxp", "cpu_raspberrypi"]
get_hw_embed_dict = get_hw_embed_dict("wmt14.en-fr")
archs_ours = {}
for device in devices:
    archs_ours[device] = []
    for s in scalarizations:
        hw_embed = get_hw_embed_dict[device]
        arch_params = metahpn(torch.tensor(s).unsqueeze(0).cuda(), torch.tensor(hw_embed).unsqueeze(0).cuda())
        arch_params = convert_arch_params_to_dict(arch_params)
        print(arch_param_to_config(arch_params, search_space))
        arch_param_discretized = discretize_arch_param(arch_params)
        arch_params_predictor = preprocess_for_predictor(arch_param_discretized, search_space)
        latency = predictor(arch_params_predictor, torch.tensor(hw_embed).unsqueeze(0).cuda())
        print(latency*200)
        archs_ours[device].append([s,arch_param_to_config(arch_params, search_space), latency.item()*200])
import pickle
pickle.dump(archs_ours, open("archs_hat.pkl", "wb"))