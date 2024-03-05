from hypernetworks.models.hpn_hat import MetaHyperNetwork
from predictors.help.loader import Data
import argparse
import torch
import os
import pickle
import numpy as np
import random
def normalization(latency, index=None, portion=0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency

def get_lat_paths_for_task(task):
    
    if task == "wmt14.en-de":
        device_paths_dict = {}
        device_paths_dict["cpu_raspberrypi"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14ende_cpu_raspberrypi.pkl"
        device_paths_dict["cpu_xeon"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14ende_cpu_xeon.pkl"
        device_paths_dict["gpu_titanxp"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14ende_gpu_titanxp.pkl"
        # load pickle files
        for device in device_paths_dict:
            with open(device_paths_dict[device], "rb") as f:
                device_paths_dict[device] = pickle.load(f)
        search_space = "space0"
    
    elif task == "wmt14.en-fr":
        device_paths_dict = {}
        device_paths_dict["cpu_raspberrypi"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14enfr_cpu_raspberrypi.pkl"
        device_paths_dict["cpu_xeon"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14enfr_cpu_xeon.pkl"
        device_paths_dict["gpu_titanxp"] = "hypernetwork_data_utils/hat/config_to_lat_list_wmt14enfr_gpu_titanxp.pkl"
        for device in device_paths_dict:
            with open(device_paths_dict[device], "rb") as f:
                # read silently
                device_paths_dict[device] = pickle.load(f)
        search_space = "space0"
    
    return device_paths_dict, search_space

def compute_lats_vector_dict(lat_datset_dicts, lat_norm):
        lats_vector_dict = {}
        for k in lat_datset_dicts.keys():
            list_archs = lat_datset_dicts[k]
            archs_lats = []
            for arch in list_archs:
                archs_lats.append((arch["latency_mean_encoder"]+arch["latency_mean_decoder"])/lat_norm)
            arch_ids_for_hw = list(range(len(archs_lats)))
            arch_inds_emb = arch_ids_for_hw[::len(arch_ids_for_hw)//10]
            lats_vector = [archs_lats[i] for i in arch_inds_emb]

            lats_vector_dict[k] = normalization(np.array(lats_vector),portion=1.0)
        return lats_vector_dict

def get_space(search_space):
    if search_space == "space1":
        d = {}
        d["encoder-embed-choice"] = [640,512]
        d["decoder-embed-choice"] = [640,512]
        d["encoder-ffn-embed-dim-choice"] = [3072, 2048, 1024, 512]
        d["decoder-ffn-embed-dim-choice"] = [3072, 2048, 1024, 512]
        d["encoder-layer-num-choice"] = [6]
        d["decoder-layer-num-choice"] = [6, 5, 4, 3, 2, 1]
        d["encoder-self-attention-heads-choice"] = [8,4,2]
        d["decoder-self-attention-heads-choice"] = [8,4,2]
        d["decoder-ende-attention-heads-choice"] = [8,4,2]
        d["decoder-arbitrary-ende-attn-choice"] = [-1, 1, 2] #[1,2,3]
        return d
    elif search_space == "space0":
        d = {}
        d["encoder-embed-choice"] = [640,512]
        d["decoder-embed-choice"] = [640,512]
        d["encoder-ffn-embed-dim-choice"] = [3072, 2048, 1024]
        d["decoder-ffn-embed-dim-choice"] = [3072, 2048, 1024]
        d["encoder-layer-num-choice"] = [6]
        d["decoder-layer-num-choice"] = [6, 5, 4, 3, 2, 1]
        d["encoder-self-attention-heads-choice"] = [8,4]
        d["decoder-self-attention-heads-choice"] = [8,4]
        d["decoder-ende-attention-heads-choice"] = [8,4]
        d["decoder-arbitrary-ende-attn-choice"] = [-1, 1, 2]
        return d

metahpn = MetaHyperNetwork(get_space("space1")).cuda()
metahpn.cuda()
p = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.]*2))
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(metahpn.parameters(), lr=1e-3)
os.makedirs("pretrained_hpns", exist_ok=True)
device_paths_dict, search_space = get_lat_paths_for_task("wmt14.en-de")
lat_norm = 200
lat_datset_dicts = get_lat_paths_for_task("wmt14.en-de")[0]
search_space = get_lat_paths_for_task("wmt14.en-de")[1]
lats_vector_dict = compute_lats_vector_dict(lat_datset_dicts, lat_norm)
devices_train = ["cpu_xeon", "gpu_titanxp"]
devices_test = ["cpu_raspberrypi"]
with open("hypernetwork_data_utils/hat//hat_lats_vector_dict_wmt14.en-de.pkl", "wb") as f:
    pickle.dump(lats_vector_dict, f)
for i in range(10):
  for j in range(1000):
    scalarization = p.sample().cuda().unsqueeze(0)
    sample_device = random.choice(devices_train)
    hw_emb = lats_vector_dict[sample_device]
    sample_hw_embed_tensor = torch.Tensor(hw_emb).cuda()
    out_encoder_embed, out_decoder_embed, out_encoder_layer, out_decoder_layer, out_encoder_ffn_embed_dim, out_decoder_ffn_embed_dim, out_encoder_attention_heads, out_decoder_attention_heads, out_decoder_ende_attention_heads, out_decoder_arbitrary_ende_attn = metahpn(scalarization, sample_hw_embed_tensor)
    random_out_encoder_embed = 1e-3*torch.randn(out_encoder_embed.shape).cuda()
    random_out_decoder_embed = 1e-3*torch.randn(out_decoder_embed.shape).cuda()
    random_out_encoder_layer = 1e-3*torch.randn(out_encoder_layer.shape).cuda()
    random_out_decoder_layer = 1e-3*torch.randn(out_decoder_layer.shape).cuda()
    random_out_encoder_ffn_embed_dim = 1e-3*torch.randn(out_encoder_ffn_embed_dim.shape).cuda()
    random_out_decoder_ffn_embed_dim = 1e-3*torch.randn(out_decoder_ffn_embed_dim.shape).cuda()
    random_out_encoder_attention_heads = 1e-3*torch.randn(out_encoder_attention_heads.shape).cuda()
    random_out_decoder_attention_heads = 1e-3*torch.randn(out_decoder_attention_heads.shape).cuda()
    random_out_decoder_ende_attention_heads = 1e-3*torch.randn(out_decoder_ende_attention_heads.shape).cuda()
    random_out_decoder_arbitrary_ende_attn = 1e-3*torch.randn(out_decoder_arbitrary_ende_attn.shape).cuda()

    loss = mse_loss(out_encoder_embed, random_out_encoder_embed) + mse_loss(out_decoder_embed, random_out_decoder_embed) + mse_loss(out_encoder_layer, random_out_encoder_layer) + mse_loss(out_decoder_layer, random_out_decoder_layer) + mse_loss(out_encoder_ffn_embed_dim, random_out_encoder_ffn_embed_dim) + mse_loss(out_decoder_ffn_embed_dim, random_out_decoder_ffn_embed_dim) + mse_loss(out_encoder_attention_heads, random_out_encoder_attention_heads) + mse_loss(out_decoder_attention_heads, random_out_decoder_attention_heads) + mse_loss(out_decoder_ende_attention_heads, random_out_decoder_ende_attention_heads) + mse_loss(out_decoder_arbitrary_ende_attn, random_out_decoder_arbitrary_ende_attn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if j%100==0:
        print("loss: ", loss)
        torch.save(metahpn.state_dict(), "pretrained_hpns/metahpn_hat_wmt14.en-de"+".pt")
        print(torch.nn.functional.softmax(out_encoder_embed, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_embed, dim=-1))
        print(torch.nn.functional.softmax(out_encoder_layer, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_layer, dim=-1))
        print(torch.nn.functional.softmax(out_encoder_ffn_embed_dim, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_ffn_embed_dim, dim=-1))
        print(torch.nn.functional.softmax(out_encoder_attention_heads, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_attention_heads, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_ende_attention_heads, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_arbitrary_ende_attn, dim=-1))



metahpn = MetaHyperNetwork(get_space("space1")).cuda()
metahpn.cuda()
p = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.]*2))
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(metahpn.parameters(), lr=1e-3)
os.makedirs("pretrained_hpns", exist_ok=True)
device_paths_dict, search_space = get_lat_paths_for_task("wmt14.en-fr")
lat_norm = 200
lat_datset_dicts = get_lat_paths_for_task("wmt14.en-fr")[0]
search_space = get_lat_paths_for_task("wmt14.en-fr")[1]
lats_vector_dict = compute_lats_vector_dict(lat_datset_dicts, lat_norm)
devices_train = ["cpu_xeon", "gpu_titanxp"]
devices_test = ["cpu_raspberrypi"]
with open("hypernetwork_data_utils/hat/hat_lats_vector_dict_wmt14.en-fr.pkl", "wb") as f:
    pickle.dump(lats_vector_dict, f)
for i in range(10):
  for j in range(1000):
    scalarization = p.sample().cuda().unsqueeze(0)
    sample_device = random.choice(devices_train)
    hw_emb = lats_vector_dict[sample_device]
    sample_hw_embed_tensor = torch.Tensor(hw_emb).cuda()
    out_encoder_embed, out_decoder_embed, out_encoder_layer, out_decoder_layer, out_encoder_ffn_embed_dim, out_decoder_ffn_embed_dim, out_encoder_attention_heads, out_decoder_attention_heads, out_decoder_ende_attention_heads, out_decoder_arbitrary_ende_attn = metahpn(scalarization, sample_hw_embed_tensor)
    random_out_encoder_embed = 1e-3*torch.randn(out_encoder_embed.shape).cuda()
    random_out_decoder_embed = 1e-3*torch.randn(out_decoder_embed.shape).cuda()
    random_out_encoder_layer = 1e-3*torch.randn(out_encoder_layer.shape).cuda()
    random_out_decoder_layer = 1e-3*torch.randn(out_decoder_layer.shape).cuda()
    random_out_encoder_ffn_embed_dim = 1e-3*torch.randn(out_encoder_ffn_embed_dim.shape).cuda()
    random_out_decoder_ffn_embed_dim = 1e-3*torch.randn(out_decoder_ffn_embed_dim.shape).cuda()
    random_out_encoder_attention_heads = 1e-3*torch.randn(out_encoder_attention_heads.shape).cuda()
    random_out_decoder_attention_heads = 1e-3*torch.randn(out_decoder_attention_heads.shape).cuda()
    random_out_decoder_ende_attention_heads = 1e-3*torch.randn(out_decoder_ende_attention_heads.shape).cuda()
    random_out_decoder_arbitrary_ende_attn = 1e-3*torch.randn(out_decoder_arbitrary_ende_attn.shape).cuda()

    loss = mse_loss(out_encoder_embed, random_out_encoder_embed) + mse_loss(out_decoder_embed, random_out_decoder_embed) + mse_loss(out_encoder_layer, random_out_encoder_layer) + mse_loss(out_decoder_layer, random_out_decoder_layer) + mse_loss(out_encoder_ffn_embed_dim, random_out_encoder_ffn_embed_dim) + mse_loss(out_decoder_ffn_embed_dim, random_out_decoder_ffn_embed_dim) + mse_loss(out_encoder_attention_heads, random_out_encoder_attention_heads) + mse_loss(out_decoder_attention_heads, random_out_decoder_attention_heads) + mse_loss(out_decoder_ende_attention_heads, random_out_decoder_ende_attention_heads) + mse_loss(out_decoder_arbitrary_ende_attn, random_out_decoder_arbitrary_ende_attn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if j%100==0:
        print("loss: ", loss)
        torch.save(metahpn.state_dict(), "pretrained_hpns/metahpn_hat_wmt14.en-fr"+".pt")
        print(torch.nn.functional.softmax(out_encoder_embed, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_embed, dim=-1))
        print(torch.nn.functional.softmax(out_encoder_layer, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_layer, dim=-1))
        print(torch.nn.functional.softmax(out_encoder_ffn_embed_dim, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_ffn_embed_dim, dim=-1))
        print(torch.nn.functional.softmax(out_encoder_attention_heads, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_attention_heads, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_ende_attention_heads, dim=-1))
        print(torch.nn.functional.softmax(out_decoder_arbitrary_ende_attn, dim=-1))
