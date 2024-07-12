from predictors.gpt.net import Net
from predictors.gpt.hw_loader import HWDataset, search_spaces, convert_config_to_one_hot
from search_spaces.gpt.utils import sample_config
import torch
import pickle
metapredictor = Net(80, 256, 4, True, 10)
dataset = HWDataset()
metapredictor.load_state_dict(torch.load("predictors/gpt/metapredictor.pt"))
metapredictor.eval()
devices = dataset.gpus
max_min_energy_stats = {}
max_min_energy_stats["max"] = {}
max_min_energy_stats["min"] = {}
for device in devices:
    max_min_energy_stats["max"][device] = 0
    max_min_energy_stats["min"][device] = 100000
for i in range(50000):
    config = sample_config(search_spaces["s"], seed=i)
    config_one_hot = convert_config_to_one_hot(config, search_spaces["s"])
    config_one_hot = torch.tensor(config_one_hot).unsqueeze(0).cuda()
    for device in devices:
        batch = dataset.sample_batch(device, 1, "train")
        hw_embed = batch[2]
        hw_embed = hw_embed.float().cuda()
        output = metapredictor(config_one_hot, hw_embed)
        output = torch.squeeze(output)
        output = output.item()
        if output > max_min_energy_stats["max"][device]:
            max_min_energy_stats["max"][device] = output
        if output < max_min_energy_stats["min"][device]:
            max_min_energy_stats["min"][device] = output
    print(max_min_energy_stats)
    with open("max_min_energy_stats.pkl","wb") as f:
        pickle.dump(max_min_energy_stats, f)
