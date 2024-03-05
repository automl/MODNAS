from predictors.help.net import MetaLearner
import torch
from predictors.help.loader import Data
from predictors.help.utils import arch_encoding_ofa
import matplotlib.pyplot as plt
import pickle
ofa_predictor = MetaLearner("ofa",True,10,100).cuda()
ofa_predictor.load_state_dict(torch.load("predictor_data_utils/ofa/ofa_predictor_modified.pt"))

def normalization(latency, index=None, portion=0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency
# zero-shot evaluation

ofa_predictor.eval()
archs = torch.load("datasets/help/ofa/ofa_archs.pt")["arch"]
archs_list = []
for i in range(len(archs)):
    archs_list.append(arch_encoding_ofa(archs[i]))
base_path_latencies = "datasets/help/ofa/latency/"
dataloader = Data("meta-test","datasets/help/ofa/","ofa", [], [], ["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_xp_1","titan_xp_32","titan_xp_64","v100_1","v100_32", "v100_64", "titan_rtx_64"], num_inner_tasks=8,
                         num_meta_train_sample=4000,
                         num_sample=10,
                         num_query=1000,
                         sampled_arch_path='datasets/help/nasbench201/arch_generated_by_metad2a.txt' )
stats ={}
for device in  ["2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_xp_1","titan_xp_32","titan_xp_64","v100_1","v100_32", "v100_64", "titan_rtx_64"]:
    hw_embed, _, _, _, _, _ = dataloader.get_task(device)
    pred_latencies = []
    true_latencies = []
    ofa_predictor = ofa_predictor.cuda()

    for i in range(len(archs_list)):
        archs_list_curr = archs_list[i].cuda().unsqueeze(0)
        #print(archs_list[i].shape)
        #print(hw_embed.shape)
        pred_latency = ofa_predictor(archs_list_curr, hw_embed.cuda())
        pred_latencies.append(pred_latency)
        #print(pred_latency)
        
        true_latencies.append(torch.load(base_path_latencies+device+".pt")[i])
        #print(true_latencies[-1])
    pred_latencies = torch.tensor(pred_latencies).cuda()
    true_latencies = torch.tensor(true_latencies).cuda()
    # compute spearman rank correlation between predicted and true latencies
    from scipy.stats import spearmanr
    print(device, "Spearman", spearmanr(pred_latencies.cpu().numpy(), true_latencies.cpu().numpy()).correlation)
    # compute kendall rank correlation between predicted and true latencies
    from scipy.stats import kendalltau
    print(device, "Kendalltau", kendalltau(pred_latencies.cpu().numpy(), true_latencies.cpu().numpy()).correlation)
    stats[device] = {}
    stats[device]["max"] = max(pred_latencies)
    stats[device]["min"] = min(pred_latencies)
    print(stats)
    with open("stats_ofa.pkl","wb") as f:
        pickle.dump(stats,f)
    # plot the predicted and true latencies
    #plt.scatter(true_latencies.cpu().numpy(), pred_latencies.cpu().numpy())
    #plt.xlabel("True Latency")
    #plt.ylabel("Predicted Latency")
    #plt.title(device)
    #plt.savefig(device+".png")
    #plt.clf()

