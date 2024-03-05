import torch
from search.utils import add_global_node
from predictors.nb201.models.predictors import  GCNHardware
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

with open("datasets/help/nasbench201/joint_benchmark.pkl","rb") as f:
  data = pickle.load(f)
architects = torch.load("datasets/help/nasbench201/architecture.pt")
str_to_idx = torch.load("datasets/help/nasbench201/str_arch2idx.pt")
idx_to_str = torch.load("dataests/help/idx_arch2str.pt")
arch_inds_for_lat = torch.load("datasets/help/nasbench201/hardware_embedding_index.pt")

def sample_random_architecture(arch=None):
    if arch is None:
       arch = np.random.choice(range(15625), size = 1)[0]
    arch_info = architects[arch]
    arch_str = idx_to_str[str(arch)]
    processed_operations = add_global_node(arch_info["operation"],ifAdj=False)#.unsqueeze(0)
    processed_adjacency = add_global_node(arch_info["adjacency_matrix"],ifAdj=True)#.unsqueeze(0)
    #print(processed_operations.shape, processed_adjacency.shape)
    return processed_operations.unsqueeze(0), processed_adjacency.unsqueeze(0), arch_str

def sample_hardware_emb(hardware_name, arch_id):
   hardware_dict = {}
   hardware_dict["fpga_latency"] = "datasets/help/nasbench201/latency/fpga.pt"
   hardware_dict["pixel3_latency"] = "datasets/help/nasbench201/latency/pixel3.pt"
   hardware_dict["raspi4_latency"] = "datasets/help/nasbench201/latency/raspi4.pt"
   hardware_dict["eyeriss_latency"] = "datasets/help/nasbench201/latency/eyeriss.pt"
   hardware_dict["pixel2_latency"] = "datasets/help/nasbench201/latency/pixel2.pt"
   hardware_dict["1080ti_1_latency"] = "datasets/help/nasbench201/latency/1080ti_1.pt"
   hardware_dict["1080ti_32_latency"] = "datasets/help/nasbench201/latency/1080ti_32.pt"
   hardware_dict["1080ti_256_latency"] = "datasets/help/nasbench201/latency/1080ti_256.pt"
   hardware_dict["2080ti_1_latency"] = "datasets/help/nasbench201/latency/2080ti_1.pt"
   hardware_dict["2080ti_32_latency"] = "datasets/help/nasbench201/latency/2080ti_32.pt"
   hardware_dict["2080ti_256_latency"] = "datasets/help/nasbench201/latency/2080ti_256.pt"
   hardware_dict["titanx_1_latency"] = "datasets/help/nasbench201/latency/titanx_1.pt"
   hardware_dict["titanx_32_latency"] = "datasets/help/nasbench201/latency/titanx_32.pt"
   hardware_dict["titanx_256_latency"] = "datasets/help/nasbench201/latency/titanx_256.pt"
   hardware_dict["titanxp_1_latency"] = "datasets/help/nasbench201/latency/titanxp_1.pt"
   hardware_dict["titanxp_32_latency"] = "datasets/help/nasbench201/latency/titanxp_32.pt"
   hardware_dict["titanxp_256_latency"] = "datasets/help/nasbench201/latency/titanxp_256.pt"
   hardware_dict["titan_rtx_1_latency"] = "datasets/help/nasbench201/latency/titan_rtx_1.pt"
   hardware_dict["titan_rtx_32_latency"] = "datasets/help/nasbench201/latency/titan_rtx_32.pt"
   hardware_dict["titan_rtx_256_latency"] = "datasets/help/nasbench201/latency/titan_rtx_256.pt"
   hardware_dict["essential_ph_1_latency"] = "datasets/help/nasbench201/latency/essential_ph_1.pt"
   hardware_dict["gold_6226_latency"] = "datasets/help/nasbench201/latency/gold_6226.pt"
   hardware_dict["gold_6240_latency"] = "datasets/help/nasbench201/latency/gold_6240.pt"
   hardware_dict["samsung_a50_latency"] = "datasets/help/nasbench201/latency/samsung_a50.pt"
   hardware_dict["samsung_s7_latency"] = "datasets/help/nasbench201/latency/samsung_s7.pt"
   hardware_dict["silver_4114_latency"] = "datasets/help/nasbench201/latency/silver_4114.pt"
   hardware_dict["silver_4210r_latency"] = "datasets/help/nasbench201/latency/silver_4210r.pt"

   select_hardware_path = hardware_dict[hardware_name]
   hardware_lats = torch.load(select_hardware_path)
   hardware_emb = [hardware_lats[i] for i in arch_inds_for_lat]
   return torch.tensor(hardware_emb).cuda(), hardware_lats[arch_id]
   
def normalization(latency, index=None, portion=0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency


def get_batch(metric="fpga_latency",dataset="cifar10",batch_size = 1, test=False):
    arch_op = []
    arch_adj = []
    latencies = []
    hw_embeddings = []
    for i in range(batch_size):
        # choose metric at random
        
        processed_operations, processed_adjacency, arch_str = sample_random_architecture()
        arch_id = str_to_idx[arch_str]
        #standardize latency
        #latency = (latency - 0.5013847351074219) / (11.692700386047363-0.5013847351074219)
        hw_embed, latency = sample_hardware_emb(metric, arch_id)
        hw_embed = normalization(hw_embed,portion=1.0)
        arch_op.append(processed_operations)
        arch_adj.append(processed_adjacency)
        latencies.append(latency)
    hw_embeddings.append(torch.tensor(hw_embed).unsqueeze(0))

    return torch.cat(arch_op).cuda(), torch.cat(arch_adj).cuda(), torch.tensor(latencies).float().cuda(), torch.cat(hw_embeddings).cuda()

if __name__=="__main__":
 parser = argparse.ArgumentParser()
 metrics_train = ['1080ti_1','1080ti_32','1080ti_256', 'silver_4114', 'silver_4210r','samsung_a50', 'pixel3','essential_ph_1','samsung_s7','titanx_1','titanx_32','titanx_256','gold_6240','titan_rtx_256','gold_6226','fpga','pixel2','raspi4','eyeriss']
 metrics_test = ['titan_rtx_256','gold_6226','fpga','pixel2','raspi4','eyeriss']
 print(len(metrics_train))

 for metric in metrics_train:
  predictor = GCNHardware(8,True,10,100).cuda()
  optimizer = torch.optim.Adam(predictor.parameters(), lr = 1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500000, eta_min = 1e-6)
  criterion = torch.nn.MSELoss()
  losses = []
  metric=metric+"_latency"
  cosine_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
  predictor.load_state_dict(torch.load("predictor_data_utils/nb201/predictor_meta_learned.pth"))
  print("Metric: ", metric)
  arch_op, arch_adj, arch_lat,hw = get_batch(batch_size=1000,test=True,metric=metric)
  pred = predictor((arch_op, arch_adj),hw.repeat(1000,1))
  pred = torch.squeeze(pred)
  # compute cosine similarity between true and predicted latencies
  cos_sim = cosine_sim(pred, arch_lat.unsqueeze(1))
  print("Cosine Similarity: ", cos_sim[0])
  # compute correlation between true and predicted latencies
  corr = np.corrcoef(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
  print("Correlation: ", corr[0][1])
  # compute spearman correlation between true and predicted latencies
  from scipy.stats import spearmanr
  corr, _ = spearmanr(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
  print("Spearman Correlation: ", corr)
  # comput kendall correlation between true and predicted latencies
  from scipy.stats import kendalltau
  corr, _ = kendalltau(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
  print("Kendall Correlation: ", corr)
  #  test
  # plot true vs predicted latencies
  #create grid  for plotting
  metric_name = metric.split("_")[0:-1]
  metric_name = "_".join(metric_name)
  plt.scatter(arch_lat.cpu().detach().numpy(), pred.cpu().detach().numpy(),c="darkblue",marker="o",s=1.5)
  plt.xlabel("True Latency")
  plt.ylabel("Predicted Latency")
  if metric_name in metrics_test:
    plt.title("Device: %s, Kendall Tau: %.3f" % (metric_name, corr),color="red")
  else:
    plt.title("Device: %s, Kendall Tau: %.3f" % (metric_name, corr))
  model_plot_name = "latency_pred_%s.pdf" % metric_name
  plt.savefig(model_plot_name, bbox_inches="tight")
  plt.clf()

