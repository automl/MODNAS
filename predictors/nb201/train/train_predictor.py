import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau

from utils import add_global_node
from predictors.nb201.models.predictors import MetaLearner
from predictors.help.loader import Data

architects = torch.load("datasets/help/nasbench201/architecture.pt")
str_to_idx = torch.load("datasets/help/nasbench201/str_arch2idx.pt")
idx_to_str = torch.load("datasets/help/nasbench201/idx_arch2str.pt")
arch_inds_for_lat = torch.load("datasets/help/nasbench201/hardware_embedding_index.pt")

def sample_random_architecture():
    arch = np.random.choice(range(15625), size = 1)[0]
    arch_info = architects[arch]
    arch_str = idx_to_str[str(arch)]
    processed_operations = add_global_node(arch_info["operation"],ifAdj=False)#.unsqueeze(0)
    processed_adjacency = add_global_node(arch_info["adjacency_matrix"],ifAdj=True)#.unsqueeze(0)
    return processed_operations.unsqueeze(0), processed_adjacency.unsqueeze(0), arch_str

def get_batch(dataloader, metric="fpga_latency", dataset="cifar10", batch_size = 1, test=False):
    arch_op = []
    arch_adj = []
    latencies = []
    hw_embeddings = []
    # choose metric at random
    hw_emb, device = dataloader.sample_device_embedding(
        mode=("meta_train" if not test else "meta_test")
    )
    metric = device + '_latency'

    for i in range(batch_size):
        processed_operations, processed_adjacency, arch_str = sample_random_architecture()
        arch_id = str_to_idx[arch_str]

        gt_latency = dataloader.latency[device][arch_id]
        arch_op.append(processed_operations)
        arch_adj.append(processed_adjacency)
        latencies.append(gt_latency)
    hw_embeddings.append(hw_emb)

    return torch.cat(arch_op).cuda(), torch.cat(arch_adj).cuda(), \
            torch.tensor(latencies).float().cuda(), torch.cat(hw_embeddings).reshape(1, -1).cuda()

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--metric", type=str, default="fpga_latency")
  args = parser.parse_args()

  predictor = MetaLearner('nasbench201',
                          hw_embed_on=True,
                          hw_embed_dim=10,
                          layer_size=100).cuda()
  optimizer = torch.optim.Adam(predictor.parameters(), lr = 1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500000, eta_min = 1e-6)
  criterion = torch.nn.MSELoss()
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

  for i in range(5000):
    for j in range(100):
       arch_op, arch_adj, arch_lat, hw = get_batch(help_loader, batch_size=64, metric=args.metric)
       pred = predictor((arch_op, arch_adj), hw)
       pred = torch.squeeze(pred)
       arch_lat = torch.squeeze(arch_lat)
       loss = criterion(pred, arch_lat)
       optimizer.zero_grad()
       loss.backward()
       # Clip gradients
       torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5)
       optimizer.step()
       optimizer.zero_grad()
       print("Epoch: ", i, "Batch: ", j, "Loss: ", loss.item())

    model_save_name = "predictor_meta_learned_correct.pth"
    torch.save(predictor.state_dict(), model_save_name)
    print("Pred: ", pred[0])
    print("Latency: ", arch_lat[0])
    print("Loss: ", loss.item())
    scheduler.step()

    # compute cosine similarity between true and predicted latencies
    cos_sim = cosine_sim(pred, arch_lat.unsqueeze(1))
    print("Cosine Similarity: ", cos_sim[0])
    # compute correlation between true and predicted latencies


    # test
    arch_op, arch_adj, arch_lat,hw = get_batch(help_loader, batch_size=1000, metric=args.metric, test=True)
    pred = predictor((arch_op, arch_adj),hw)
    # plot true vs predicted latencies
    pred = torch.squeeze(pred)
    corr = np.corrcoef(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
    print("Correlation: ", corr[0][1])
    # compute spearman correlation between true and predicted latencies
    corr, _ = spearmanr(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
    print("Spearman Correlation: ", corr)
    # comput kendall correlation between true and predicted latencies
    corr, _ = kendalltau(pred.cpu().detach().numpy(), arch_lat.cpu().detach().numpy())
    print("Kendall Correlation: ", corr)
    plt.scatter(arch_lat.cpu().detach().numpy(), pred.cpu().detach().numpy())
    plt.xlabel("True Latency")
    plt.ylabel("Predicted Latency")

    # save
    model_plot_name = "true_vs_pred.png"
    plt.savefig(model_plot_name)
    plt.clf()
