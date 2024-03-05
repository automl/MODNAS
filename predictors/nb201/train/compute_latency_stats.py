import os
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau

from search.utils import add_global_node
from predictors.nb201.models.predictors import MetaLearner
from predictors.help.loader import Data

architects = torch.load("datasets/help/nasbench201/architecture.pt")
str_to_idx = torch.load("datasets/help/nasbench201/str_arch2idx.pt")
idx_to_str = torch.load("datasets/help/nasbench201/idx_arch2str.pt")
arch_inds_for_lat = torch.load("datasets/help/nasbench201/hardware_embedding_index.pt")

def sample_random_architecture():
    arch = np.random.choice(range(15625), size = 1)[0]
    return arch

def eval_architecture(arch):
    arch_info = architects[arch]
    arch_str = idx_to_str[str(arch)]
    processed_operations = add_global_node(arch_info["operation"],ifAdj=False)#.unsqueeze(0)
    processed_adjacency = add_global_node(arch_info["adjacency_matrix"],ifAdj=True)#.unsqueeze(0)
    return processed_operations.unsqueeze(0), processed_adjacency.unsqueeze(0), arch_str

def get_batch(dataloader, metric="fpga_latency", dataset="cifar10",
              batch_size=1, test=False, eval_all=False):
    arch_op = []
    arch_adj = []
    latencies = []
    hw_embeddings = []
    # choose metric at random
    hw_emb, device = dataloader.sample_device_embedding(
        mode=("meta_train" if not test else "meta_test")
    )
    metric = device + '_latency'
    batch_size = batch_size if not eval_all else 15625

    for i in range(batch_size):
        arch = i if eval_all else sample_random_architecture()
        processed_operations, processed_adjacency, arch_str = eval_architecture(arch)
        arch_id = str_to_idx[arch_str]
        gt_latency = dataloader.latency[device][arch_id]
        arch_op.append(processed_operations)
        arch_adj.append(processed_adjacency)
        latencies.append(gt_latency)
        hw_embeddings.append(hw_emb)

    return torch.cat(arch_op), torch.cat(arch_adj), \
            torch.tensor(latencies).float(), torch.cat(hw_embeddings).reshape(batch_size, -1)
def get_all_archs(dataloader, metric="fpga_latency", dataset="cifar10",
              batch_size=1, test=False, eval_all=False):
    arch_op = []
    arch_adj = []
    latencies = []
    hw_embeddings = []
    # choose metric at random
    hw_emb, device = dataloader.sample_specific_device_embedding(device=metric)
    metric = device + '_latency'
    batch_size = 15625
    for i in range(batch_size):
        arch = i 
        processed_operations, processed_adjacency, arch_str = eval_architecture(arch)
        arch_id = str_to_idx[arch_str]
        gt_latency = dataloader.latency[device][arch_id]
        arch_op.append(processed_operations)
        arch_adj.append(processed_adjacency)
        latencies.append(gt_latency)
    hw_embeddings.append(hw_emb)
    print(hw_emb.shape)
    return torch.cat(arch_op), torch.cat(arch_adj), \
            torch.tensor(latencies).float(), hw_emb.float()

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--save_path", type=str, default="help_pred")
  parser.add_argument("--load_path", type=str,
                      default="predictor_data_utils/nb201/predictor_meta_learned.pth")
  # 'predictor_meta_learned.pth'
  args = parser.parse_args()

  predictor = MetaLearner('nasbench201',
                          hw_embed_on=True,
                          hw_embed_dim=10,
                          layer_size=100)
  if "help_max_corr.pt" in args.load_path:
    predictor.load_state_dict(torch.load(args.load_path)['model'])
  else:
    predictor.load_state_dict(torch.load(args.load_path))

  losses = []
  cosine_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
  help_loader = Data(mode='meta_test',
                     data_path='datasets/help/nasbench201',
                     search_space='nasbench201',
                     meta_train_devices=['1080ti_1',
                                         '1080ti_32',
                                         '1080ti_256',
                                         'silver_4114',
                                         'silver_4210r',
                                         'samsung_a50',
                                         'pixel3',
                                         'essential_ph_1',
                                         'samsung_s7'],
                     meta_valid_devices=['titanx_1',
                                         'titanx_32',
                                         'titanx_256',
                                         'gold_6240'],
                     meta_test_devices=['titan_rtx_256',
                                        'gold_6226',
                                        'fpga',
                                        'pixel2',
                                        'raspi4',
                                        'eyeriss'],
                     num_inner_tasks=8,
                     num_meta_train_sample=900,
                     num_sample=10,
                     num_query=1000,
                     sampled_arch_path=\
                     'datasets/help/nasbench201/arch_generated_by_metad2a.txt'
                    )


  stats = {}
  for metric in (help_loader.meta_test_devices+help_loader.meta_train_devices+help_loader.meta_valid_devices):
    print("Metric: ", metric)
    # test
    arch_op, arch_adj, arch_lat, hw = get_all_archs(help_loader, metric=metric, batch_size=15625, test=True, eval_all=True)
    # back size too large iterate over chunks
    predictions = []
    start = 0
    end = 1000
    while(end <= 15625):
        pred = predictor((arch_op[start:end], arch_adj[start:end]), hw)
        predictions.extend([a.item() for a in pred])
        print(len(predictions))
        start = end
        if end == 15625:
            break
        if end + 1000 > 15625:
            end = 15625
        else:
            end += 1000
    # process last chunk
    pred = torch.tensor(predictions)
    #pred = predictor((arch_op, arch_adj), hw)
    #arch_lat = arch_lat.unsqueeze(1)
    # compute cosine similarity between true and predicted latencies
    #cos_sim = cosine_sim(pred, arch_lat)
    #print("Cosine Similarity: ", cos_sim[0])
    ## compute correlation between true and predicted latencies
    #corr = np.corrcoef(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
    #print("Correlation: ", corr[0][1])
    # compute spearman correlation between true and predicted latencies
    corr, _ = spearmanr(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
    print("Spearman Correlation: ", corr)
    # comput kendall correlation between true and predicted latencies
    corr, _ = kendalltau(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
    print("Kendall Correlation: ", corr)

    # plot true vs predicted latencies
    plt.scatter(arch_lat.cpu().detach().numpy(), pred.cpu().detach().numpy())
    plt.title(f"{metric}: {corr}")
    plt.xlabel("True Latency")
    plt.ylabel("Predicted Latency")
    # save max, min and mean latencies for each metric
    max = np.max(pred.cpu().detach().numpy())
    min = np.min(pred.cpu().detach().numpy())
    mean = np.mean(pred.cpu().detach().numpy())
    std = np.std(pred.cpu().detach().numpy())
    print("Max: ", max)
    print("Min: ", min)
    print("Mean: ", mean)
    stats[metric+"_latency"] = {}
    stats[metric+"_latency"]["max"] = max
    stats[metric+"_latency"]["min"] = min
    stats[metric+"_latency"]["mean"] = mean
    stats[metric+"_latency"]["std"] = std
    # save
    save_path = f'predictor_data_utils/nb201/'

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "all_stats.pkl"), 'wb') as f:
       pickle.dump(stats, f)
    model_plot_name = "scatter_{}.png".format(metric)
    plt.savefig(os.path.join(save_path, model_plot_name))
    plt.clf()

