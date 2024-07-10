from syne_tune import Reporter
import pickle
import torch
report = Reporter()
from search_spaces.nb201.model_search import NASBench201SearchSpace
def get_max_min(metric, data):
    all_arch_metrics = []
    for arch in data.keys():
        all_arch_metrics.append(data[arch][metric])
    return max(all_arch_metrics), min(all_arch_metrics)

def get_max_min_error(dataset, data):
    all_arch_metrics = []
    for arch in data.keys():
        all_arch_metrics.append(100-data[arch][dataset])
    return max(all_arch_metrics), min(all_arch_metrics)

def convert_arch_to_arch_param(arch):
    print(arch)
    arch_param = torch.zeros([6, 5])
    for i in range(6):
        arch_param[i,arch["edge"+str(i)]] = 1
    return arch_param

def objective(arch, metric, dataset):
    arch_param =convert_arch_to_arch_param(arch)
    model = NASBench201SearchSpace(16, 5, 4, 10)#.cuda()
    model.set_arch_params(arch_param)
    arch_str = model.genotype().tostr()

    with open("/work/dlclarge1/sukthank-modnas/MODNAS_ICML/HELP/benchmark_all_hw_metrics.pkl", "rb") as f:
         data = pickle.load(f)
    error = 100-data[arch_str][dataset]
    latency = data[arch_str][metric]
    # normalize error
    max_error, min_error = get_max_min_error(dataset, data)
    error = (error-min_error)/(max_error-min_error)
    # normalize latency
    max_latency, min_latency = get_max_min(metric, data)
    latency = (latency-min_latency)/(max_latency-min_latency)
    report(error = error, latency = latency)

if __name__ == "__main__":
    import logging
    import argparse
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    num_edges = 6
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--st_checkpoint_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--metric", type=str, required=True)
    for i in range(num_edges):
        parser.add_argument(f"--edge{i}", type=int, default=0)
    args = parser.parse_args()
    arch = {}
    for i in range(num_edges):
        arch["edge"+str(i)] = getattr(args, f"edge{i}")
    objective(arch, args.metric, "cifar10")
    #print(objective(arch, args.metric, "cifar10"))