import json
import logging
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import json
import logging
import random
from pathlib import Path
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from pygmo import hypervolume
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

from predictors.help.utils import denorm
from optimizers.help import run_nas, get_parser
from predictors.help.loader import Data



class CustomCosineAnnealing():
    def __init__(self, eta_min, eta_max, T_max):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_max = T_max

    def get_value(self, epoch):
        return (self.eta_min + (self.eta_max - self.eta_min) *
                (1 + math.cos(math.pi * epoch / self.T_max)) / 2)

def compute_l2_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def max_min_norm(x,min_val,max_val):
    return (x - min_val) / (max_val - min_val)

def normalization(latency, index=None, portion=0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency

def run_metad2a_help(rays, device, help_loader, seed=1):
    help_args = get_parser()
    help_args.nas_target_device = device
    help_args.seed = seed

    latencies = help_loader.latency[device]
    archs = list()
    for s1 in rays[:, 1]:
        latency_constr = denorm(s1, max(latencies), min(latencies))
        print("Latency constraint: ", latency_constr)
        help_args.latency_constraint = latency_constr
        acc, lat, arch = run_nas(help_args)
        if arch is not None:
            archs.append(arch)

    return archs



def run_ofa_search_help(rays, device, help_loader, seed=1):
    help_args = get_parser()
    help_args.nas_target_device = device
    help_args.seed = seed
    accs = []
    lats = []
    latencies = help_loader.latency[device]
    archs = list()
    for s1 in rays[:, 1]:
        latency_constr = denorm(s1, max(latencies), min(latencies))
        help_args.latency_constraint = latency_constr
        acc, lat, arch = run_nas(help_args)
        if arch is not None:
            archs.append(arch)
        accs.append(acc)
        lats.append(lat)

    return archs, accs, lats

def get_path_to_pretrained_hpns(name):
    return "hypernetwork_data_utils/nb201/{}.pth".format(name)

def get_hardware_dict():
   hardware_dict = {}
   hardware_dict["fpga_latency"] = "datasets/help//nasbench201/latency/fpga.pt"
   hardware_dict["pixel3_latency"] = "datasets/help//nasbench201/latency/pixel3.pt"
   hardware_dict["raspi4_latency"] = "datasets/help//nasbench201/latency/raspi4.pt"
   hardware_dict["eyeriss_latency"] = "datasets/help//nasbench201/latency/eyeriss.pt"
   hardware_dict["pixel2_latency"] = "datasets/help//nasbench201/latency/pixel2.pt"
   hardware_dict["1080ti_1_latency"] = "datasets/help//nasbench201/latency/1080ti_1.pt"
   hardware_dict["1080ti_32_latency"] = "datasets/help//nasbench201/latency/1080ti_32.pt"
   hardware_dict["1080ti_256_latency"] = "datasets/help//nasbench201/latency/1080ti_256.pt"
   hardware_dict["2080ti_1_latency"] = "datasets/help//nasbench201/latency/2080ti_1.pt"
   hardware_dict["2080ti_32_latency"] = "datasets/help//nasbench201/latency/2080ti_32.pt"
   hardware_dict["2080ti_256_latency"] = "datasets/help//nasbench201/latency/2080ti_256.pt"
   hardware_dict["titanx_1_latency"] = "datasets/help//nasbench201/latency/titanx_1.pt"
   hardware_dict["titanx_32_latency"] = "datasets/help//nasbench201/latency/titanx_32.pt"
   hardware_dict["titanx_256_latency"] = "datasets/help//nasbench201/latency/titanx_256.pt"
   hardware_dict["titanxp_1_latency"] = "datasets/help//nasbench201/latency/titanxp_1.pt"
   hardware_dict["titanxp_32_latency"] = "datasets/help//nasbench201/latency/titanxp_32.pt"
   hardware_dict["titanxp_256_latency"] = "datasets/help//nasbench201/latency/titanxp_256.pt"
   hardware_dict["titan_rtx_1_latency"] = "datasets/help//nasbench201/latency/titan_rtx_1.pt"
   hardware_dict["titan_rtx_32_latency"] = "datasets/help//nasbench201/latency/titan_rtx_32.pt"
   hardware_dict["titan_rtx_256_latency"] = "datasets/help//nasbench201/latency/titan_rtx_256.pt"
   hardware_dict["essential_ph_1_latency"] = "datasets/help//nasbench201/latency/essential_ph_1.pt"
   hardware_dict["gold_6226_latency"] = "datasets/help//nasbench201/latency/gold_6226.pt"
   hardware_dict["gold_6240_latency"] = "datasets/help//nasbench201/latency/gold_6240.pt"
   hardware_dict["samsung_a50_latency"] = "datasets/help//nasbench201/latency/samsung_a50.pt"
   hardware_dict["samsung_s7_latency"] = "datasets/help//nasbench201/latency/samsung_s7.pt"
   hardware_dict["silver_4114_latency"] = "datasets/help//nasbench201/latency/silver_4114.pt"
   hardware_dict["silver_4210r_latency"] = "datasets/help//nasbench201/latency/silver_4210r.pt"
   return hardware_dict

# declare global variables
def sample_hardware_emb(hardware_name):
   hardware_dict = get_hardware_dict()
   arch_inds_for_lat = torch.load("datasets/help//nasbench201/hardware_embedding_index.pt")

   select_hardware_path = hardware_dict[hardware_name]
   hardware_lats = torch.load(select_hardware_path)
   hardware_emb = [hardware_lats[i] for i in arch_inds_for_lat]
   return torch.tensor(hardware_emb).cuda()

def sample_all_devices():
   hardware_dict = get_hardware_dict()
   arch_inds_for_lat = torch.load("datasets/help//nasbench201/hardware_embedding_index.pt")
   hw_embeddings = []
   metrics = []
   for hardware_name in hardware_dict.keys():
       select_hardware_path = hardware_dict[hardware_name]
       hardware_lats = torch.load(select_hardware_path)
       hardware_emb = [hardware_lats[i] for i in arch_inds_for_lat]
       hardware_emb = torch.tensor(hardware_emb).cuda()
       hw_embeddings.append(hardware_emb)
       metrics.append(hardware_name)
   return hw_embeddings, metrics

def sample_device_embedding(mode="train"):
    if mode == "train":
       metrics = ["fpga_latency", "pixel3_latency", "raspi4_latency", "1080ti_32_latency", "1080ti_256_latency", "2080ti_1_latency",  "2080ti_256_latency", "titanx_1_latency", "titanx_32_latency", "titanx_256_latency", "titanxp_1_latency", "titanxp_32_latency", "titanxp_256_latency", "titan_rtx_1_latency", "titan_rtx_32_latency", "titan_rtx_256_latency", "essential_ph_1_latency", "gold_6226_latency", "gold_6240_latency", "samsung_a50_latency", "samsung_s7_latency", "silver_4114_latency", "silver_4210r_latency"]
    else:
       metrics = ["eyeriss_latency", "pixel2_latency", "1080ti_1_latency","2080ti_32_latency"]
    metric = np.random.choice(metrics, size = 1)[0]
    return sample_hardware_emb(metric), metric

def filter_data(errors, latencies, archs):
    errors = np.array(errors)
    errors_filtered = errors[errors < 90]
    latencies_filtered = np.array(latencies)[errors < 90]
    archs_filtered = np.array(archs)[errors < 90]

    return errors_filtered, latencies_filtered, archs_filtered

def get_pareto_front(Xs, Ys, archs, maxX=False, maxY=False):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i], archs[i]] for i in range(
        len(Xs))], reverse=maxY, key=lambda element: (element[0], element[1]))
    pareto_front = [[sorted_list[0][0], sorted_list[0][1]]]
    archs_pareto = [sorted_list[0][2]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
                archs_pareto.append(pair[2])

        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
                archs_pareto.append(pair[2])

    print("Architecures in Pareto Frontier: ", archs_pareto)
    return pareto_front, archs_pareto


def get_sotl_data(data, model, archs, dataset, metric, maxX=False, maxY=False,
                  sotl_epochs=3, min_err=0, max_err=100, min_lat=0, max_lat=100):
    errors, latencies = list(), list()
    sotl_list = list()

    for k in archs:
        errors.append(100 - data[k][dataset])
        latencies.append(data[k][metric])

        arch_id = model.api.archstr2index[k]
        train_losses = list(
             model.api.query_by_index(
                 arch_id
             ).all_results[('cifar10-valid',777)].train_losses.values()
        )
        ema_sotl = compute_ema_sotl(train_losses, epochs=sotl_epochs)
        sotl_list.append(ema_sotl)

    errors_filtered, latencies_filtered, archs_filtered = filter_data(errors, latencies, archs)

    # filter the sotl_list first
    _, sotl_list, _ = filter_data(errors, sotl_list, archs)
    # select pareto front architectures using SoTL-EMA
    _, sotl_archs = get_pareto_front(sotl_list,
                                     latencies_filtered,
                                     archs_filtered,
                                     maxX=False, maxY=False)

    # query the results for the selected architectures and plot the pareto
    # front
    sotl_errors, sotl_latencies = list(), list()

    for k in sotl_archs:
        sotl_errors.append(100 - data[k][dataset])
        sotl_latencies.append(data[k][metric])

    errors_filtered = max_min_norm(errors_filtered, min_err, max_err)
    latencies_filtered = max_min_norm(latencies_filtered, min_lat, max_lat)
    sotl_errors = max_min_norm(sotl_errors, min_err, max_err)
    sotl_latencies = max_min_norm(sotl_latencies, min_lat, max_lat)

    return errors_filtered, latencies_filtered, archs_filtered, sotl_errors, sotl_latencies, sotl_archs

def get_data_from_pareto(pareto_front):
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    return pf_X, pf_Y

def plot_pareto_frontier(Xs, Ys, archs, maxX=False, maxY=False, color="blue",
                         xlabel="", ylabel="", marker='.', linestyle='-',
                         legend="", scatter=False):
    '''Plotting process'''
    if scatter:
        plt.scatter(Xs, Ys, color=color, marker=marker)
    pareto_front, archs_pareto = get_pareto_front(Xs, Ys, archs, maxX, maxY)

    pf_X, pf_Y = get_data_from_pareto(pareto_front)

    plt.scatter(pf_X, pf_Y, color=color, marker=marker)
    plt.plot(pf_X, pf_Y, color=color, label=legend, marker=marker,
             linestyle=linestyle)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.legend()
    # plt.show()
    return pareto_front, archs_pareto

def update_metrics_dict(metrics_dict, errors, latencies, true_errors,
                        true_latencies, all_errors, archs, archs_true,
                        method_prefix='true', metric='fpga_latency', rays=None):

    metrics = np.array([errors, latencies]).T
    metrics_true = np.array([true_errors, true_latencies]).T
    metrics_dict[f"{metric}_hypervolume_{method_prefix}"] = compute_hypervolume(metrics, [1, 1])

    ind = GD(metrics_true)
    metrics_dict[f"{metric}_gd_{method_prefix}"] = ind(metrics)

    ind = GDPlus(metrics_true)
    metrics_dict[f"{metric}_gd_plus_{method_prefix}"] = ind(metrics)

    ind = IGD(metrics_true)
    metrics_dict[f"{metric}_igd_{method_prefix}"] = ind(metrics)

    ind = IGDPlus(metrics_true)
    metrics_dict[f"{metric}_igd_plus_{method_prefix}"] = ind(metrics)

    metrics_dict[f"{metric}_noise_error_{method_prefix}"] = np.std(np.array(errors))
    metrics_dict[f"{metric}_noise_latency_{method_prefix}"] = np.std(np.array(latencies))
    metrics_dict[f"{metric}_off_pareto_{method_prefix}"] = len(all_errors) - len(errors)
    metrics_dict[f"{metric}_intersection_with_true_{method_prefix}"] = len(
        set(archs_true).intersection(set(archs))
    )

    # cosine similarity between rays and predictions
    #if rays is not None:
        #cosine_sim = nn.CosineSimilarity(dim=-1,
                                         #eps=1e-6)(torch.tensor(metrics),
                                                   #torch.tensor(rays)).mean().numpy()
        #metrics_dict[f"{metric}_cosine_sim_{method_prefix}"] = cosine_sim

# Create an ensemble prediction by averaging the predictions
def get_ensemble_data(hpn_load_path, model, device, epoch, save_path,
                      dataset, seed=1, rays=None, help_loader=None):
    def hpn_ensemble_predict(hpn_models, scalarization, hw_emb):
        predictions = []
        for model in hpn_models:
            model.eval()
            with torch.no_grad():
                outputs = model(scalarization, hw_emb)
                predictions.append(outputs.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0).mean(dim=0)
        return predictions

    from hypernetworks.hpns_nb201 import MetaHyperNetwork
    # Load ensemble baselearners
    hpn_model_list = []
    for hpn_path in hpn_load_path:
        hpn = MetaHyperNetwork(num_random_devices=50,
                               use_zero_init=False,
                               use_softmax=True)
        full_hpn_path = os.path.join(hpn_path, "hpn.pt")
        hpn.load_state_dict(torch.load(full_hpn_path))
        hpn = hpn.cuda()
        hpn_model_list.append(hpn)
    print("Loaded hpns from ", hpn_load_path)

    hw_embedding, _ = help_loader.sample_specific_device_embedding(device=device)
    hw_embedding = hw_embedding.cuda()
    archs_help = []
    for point in rays:
        scalarization = torch.tensor([point[0], point[1]]).cuda()
        # convert to float
        scalarization = scalarization.float()
        arch_params = hpn_ensemble_predict(hpn_model_list, scalarization.unsqueeze(0),
                                           hw_embedding.unsqueeze(0))
        model.set_arch_params(arch_params)
        genotype = model.genotype().tostr()
        archs_help.append(genotype)

    metric = device + '_latency'
    with open("datasets/help/nasbench201/benchmark_all_hw_metrics.pkl", "rb") as f:
         data = pickle.load(f)

    true_errors = []
    true_latencies = []
    archs_true = []

    for k in data.keys():
        true_errors.append(100-data[k][dataset])
        true_latencies.append(data[k][metric])
        archs_true.append(k)

    # filter errors and latencies to have error less than 90%
    true_errors_filtered, true_latencies_filtered, archs_true_filtered = \
            filter_data(true_errors, true_latencies, archs_true)

    max_lat, min_lat = max(true_latencies_filtered), min(true_latencies_filtered)
    max_err, min_err = max(true_errors_filtered), min(true_errors_filtered)
    # normalize max_min 
    true_errors_filtered = max_min_norm(true_errors_filtered, min_err, max_err)
    true_latencies_filtered = max_min_norm(true_latencies_filtered, min_lat, max_lat)

    pareto_front_true = get_pareto_front(true_errors_filtered,
                                         true_latencies_filtered,
                                         archs_true_filtered, maxX=False,
                                         maxY=False)
    p_true_errors, p_true_latencies = get_data_from_pareto(pareto_front_true[0])

    metrics_dict = dict()
    metrics_dict[f"{metric}_hypervolume_true"] = compute_hypervolume(
        np.array([p_true_errors, p_true_latencies]).T, [1,1])

    help_errors_filtered, help_latencies_filtered, help_archs_filtered, help_sotl_errors, help_sotl_latencies, help_sotl_archs = \
            get_sotl_data(data, model, archs_help, dataset, metric,
                          False, False, 12,
                          min_err, max_err, min_lat, max_lat)

    pareto_front_help = get_pareto_front(help_errors_filtered,
                                         help_latencies_filtered,
                                         help_archs_filtered, maxX=False,
                                         maxY=False)
    pareto_front_help_sotl = get_pareto_front(help_sotl_errors,
                                              help_sotl_latencies,
                                              help_sotl_archs, maxX=False,
                                              maxY=False)

    p_help_errors, p_help_latencies = get_data_from_pareto(pareto_front_help[0])
    p_help_sotl_errors, p_help_sotl_latencies = get_data_from_pareto(pareto_front_help_sotl[0])
    update_metrics_dict(metrics_dict, p_help_errors, p_help_latencies, p_true_errors,
                        p_true_latencies, help_errors_filtered, pareto_front_help[1],
                        archs_true_filtered, 'hpn_ensemble', metric, rays)
    update_metrics_dict(metrics_dict, p_help_sotl_errors, p_help_sotl_latencies, p_true_errors,
                        p_true_latencies, help_sotl_errors, pareto_front_help_sotl[1],
                        archs_true_filtered, 'hpn_ensemble_sotl', metric, rays)

    print("Metrics: ", metrics_dict)
    save_name = "metrics_ensemble" + ".pkl"
    save_path = save_path + "/ensemble/{}".format(metric)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, save_name)
    with open(save_path, "wb") as f:
        pickle.dump(metrics_dict, f)
    return metrics_dict


def get_meta_d2a_data(model, metric, epoch, save_path,
                      dataset, seed=1, rays=None, help_loader=None):

    # get MetaD2A + HELP architectures
    if help_loader is not None:
        archs_help = run_metad2a_help(rays, metric, help_loader, seed=seed)

    metric += '_latency'
    with open("datasets/help/nasbench201/benchmark_all_hw_metrics.pkl", "rb") as f:
         data = pickle.load(f)

    true_errors = []
    true_latencies = []
    archs_true = []

    for k in data.keys():
        true_errors.append(100-data[k][dataset])
        true_latencies.append(data[k][metric])
        archs_true.append(k)

    # filter errors and latencies to have error less than 90%
    true_errors_filtered, true_latencies_filtered, archs_true_filtered = \
            filter_data(true_errors, true_latencies, archs_true)

    max_lat, min_lat = max(true_latencies_filtered), min(true_latencies_filtered)
    max_err, min_err = max(true_errors_filtered), min(true_errors_filtered)
    # normalize max_min 
    true_errors_filtered = max_min_norm(true_errors_filtered, min_err, max_err)
    true_latencies_filtered = max_min_norm(true_latencies_filtered, min_lat, max_lat)

    pareto_front_true = get_pareto_front(true_errors_filtered,
                                         true_latencies_filtered,
                                         archs_true_filtered, maxX=False,
                                         maxY=False)
    p_true_errors, p_true_latencies = get_data_from_pareto(pareto_front_true[0])

    metrics_dict = dict()
    metrics_dict[f"{metric}_hypervolume_true"] = compute_hypervolume(
        np.array([p_true_errors, p_true_latencies]).T, [1,1])

    help_errors_filtered, help_latencies_filtered, help_archs_filtered, help_sotl_errors, help_sotl_latencies, help_sotl_archs = \
            get_sotl_data(data, model, archs_help, dataset, metric,
                          False, False, 12,
                          min_err, max_err, min_lat, max_lat)

    pareto_front_help = get_pareto_front(help_errors_filtered,
                                         help_latencies_filtered,
                                         help_archs_filtered, maxX=False,
                                         maxY=False)
    pareto_front_help_sotl = get_pareto_front(help_sotl_errors,
                                              help_sotl_latencies,
                                              help_sotl_archs, maxX=False,
                                              maxY=False)

    p_help_errors, p_help_latencies = get_data_from_pareto(pareto_front_help[0])
    p_help_sotl_errors, p_help_sotl_latencies = get_data_from_pareto(pareto_front_help_sotl[0])
    update_metrics_dict(metrics_dict, p_help_errors, p_help_latencies, p_true_errors,
                        p_true_latencies, help_errors_filtered, pareto_front_help[1],
                        archs_true_filtered, 'help', metric, rays)
    update_metrics_dict(metrics_dict, p_help_sotl_errors, p_help_sotl_latencies, p_true_errors,
                        p_true_latencies, help_sotl_errors, pareto_front_help_sotl[1],
                        archs_true_filtered, 'help_sotl', metric, rays)

    print("Metrics: ", metrics_dict)
    save_name = "metrics_help" + ".pkl"
    save_path = save_path + "/statistics/{}".format(metric)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, save_name)
    with open(save_path, "wb") as f:
        pickle.dump(metrics_dict, f)
    return metrics_dict


def analysis(archs, archs_random, archs_random_hpn, model, metric, epoch,
             dataset, args, rays=None, help_loader=None):

    # get MetaD2A + HELP architectures
    if help_loader is not None:
        archs_help = run_metad2a_help(rays, metric, help_loader)

    metric += '_latency'
    with open("datasets/help/nasbench201/benchmark_all_hw_metrics.pkl", "rb") as f:
         data = pickle.load(f)

    true_errors = []
    true_latencies = []
    archs_true = []

    for k in data.keys():
        true_errors.append(100-data[k][dataset])
        true_latencies.append(data[k][metric])
        archs_true.append(k)

    # filter errors and latencies to have error less than 90%
    true_errors_filtered, true_latencies_filtered, archs_true_filtered = \
            filter_data(true_errors, true_latencies, archs_true)

    max_lat, min_lat = max(true_latencies_filtered), min(true_latencies_filtered)
    max_err, min_err = max(true_errors_filtered), min(true_errors_filtered)
    # normalize max_min 
    true_errors_filtered = max_min_norm(true_errors_filtered, min_err, max_err)
    true_latencies_filtered = max_min_norm(true_latencies_filtered, min_lat, max_lat)


    # Our method data
    predicted_errors_filtered, predicted_latencies_filtered, archs_filtered, sotl_errors, sotl_latencies, sotl_archs = \
            get_sotl_data(data, model, archs, dataset, metric, False, False, 12,
                          min_err, max_err, min_lat, max_lat)
    # RS data
    rs_errors_filtered, rs_latencies_filtered, rs_archs_filtered, rs_sotl_errors, rs_sotl_latencies, rs_sotl_archs = \
            get_sotl_data(data, model, archs_random, dataset, metric, False,
                          False, 12,
                          min_err, max_err, min_lat, max_lat)
    # Random HPN data
    rhpn_errors_filtered, rhpn_latencies_filtered, rhpn_archs_filtered, rhpn_sotl_errors, rhpn_sotl_latencies, rhpn_sotl_archs = \
            get_sotl_data(data, model, archs_random_hpn, dataset, metric,
                          False, False, 12,
                          min_err, max_err, min_lat, max_lat)

    pareto_front_true = plot_pareto_frontier(true_errors_filtered,
                                             true_latencies_filtered,
                                             archs_true_filtered, maxX=False,
                                             maxY=False, color="black",
                                             xlabel="Error", ylabel=metric,
                                             marker='.', linestyle=':',
                                             legend="Pareto Frontier")
    pareto_front_pred = plot_pareto_frontier(predicted_errors_filtered,
                                             predicted_latencies_filtered,
                                             archs_filtered, maxX=False,
                                             maxY=False, color="red",
                                             xlabel="Error", ylabel=metric,
                                             marker='h', linestyle='-.',
                                             legend="{}".format(args.optimizer_type),
                                             scatter=False)
    pareto_front_sotl = plot_pareto_frontier(sotl_errors,
                                             sotl_latencies,
                                             sotl_archs, maxX=False,
                                             maxY=False, color="magenta",
                                             xlabel="Error", ylabel=metric,
                                             marker='^', linestyle='-',
                                             legend="{} (SoTL-EMA)".format(args.optimizer_type),
                                             scatter=True)
    pareto_front_rs = plot_pareto_frontier(rs_errors_filtered,
                                           rs_latencies_filtered,
                                           rs_archs_filtered, maxX=False,
                                           maxY=False, color="blue",
                                           xlabel="Error", ylabel=metric,
                                           marker='+', linestyle='-.',
                                           legend="RS",
                                           scatter=False)
    pareto_front_rs_sotl = plot_pareto_frontier(rs_sotl_errors,
                                                rs_sotl_latencies,
                                                rs_sotl_archs, maxX=False,
                                                maxY=False, color="cyan",
                                                xlabel="Error", ylabel=metric,
                                                marker='+', linestyle='-',
                                                legend="RS (SoTL-EMA)",
                                                scatter=True)
    pareto_front_rhpn = plot_pareto_frontier(rhpn_errors_filtered,
                                             rhpn_latencies_filtered,
                                             rhpn_archs_filtered, maxX=False,
                                             maxY=False, color="orange",
                                             xlabel="Error", ylabel=metric,
                                             marker='s', linestyle='-.',
                                             legend="RandHPN",
                                             scatter=False)
    pareto_front_rhpn_sotl = plot_pareto_frontier(rhpn_sotl_errors,
                                                  rhpn_sotl_latencies,
                                                  rhpn_sotl_archs, maxX=False,
                                                  maxY=False, color="gold",
                                                  xlabel="Error", ylabel=metric,
                                                  marker='s', linestyle='-',
                                                  legend="RandHPN (SoTL-EMA)",
                                                  scatter=True)

    p_true_errors, p_true_latencies = get_data_from_pareto(pareto_front_true[0])
    p_predicted_errors, p_predicted_latencies = get_data_from_pareto(pareto_front_pred[0])
    p_sotl_errors, p_sotl_latencies = get_data_from_pareto(pareto_front_sotl[0])
    p_rs_errors, p_rs_latencies = get_data_from_pareto(pareto_front_rs[0])
    p_rs_sotl_errors, p_rs_sotl_latencies = get_data_from_pareto(pareto_front_rs_sotl[0])
    p_rhpn_errors, p_rhpn_latencies = get_data_from_pareto(pareto_front_rhpn[0])
    p_rhpn_sotl_errors, p_rhpn_sotl_latencies = get_data_from_pareto(pareto_front_rhpn_sotl[0])

    metrics_dict = dict()
    metrics_dict[f"{metric}_hypervolume_true"] = compute_hypervolume(
        np.array([p_true_errors, p_true_latencies]).T, [1,1])
    update_metrics_dict(metrics_dict, p_predicted_errors, p_predicted_latencies, p_true_errors,
                        p_true_latencies, predicted_errors_filtered, pareto_front_pred[1],
                        archs_true_filtered, 'predicted', metric, rays)
    update_metrics_dict(metrics_dict, p_sotl_errors, p_sotl_latencies, p_true_errors,
                        p_true_latencies, sotl_errors, pareto_front_sotl[1],
                        archs_true_filtered, 'sotl', metric, rays)
    update_metrics_dict(metrics_dict, p_rs_errors, p_rs_latencies, p_true_errors,
                        p_true_latencies, rs_errors_filtered, pareto_front_rs[1],
                        archs_true_filtered, 'rs', metric, rays)
    update_metrics_dict(metrics_dict, p_rs_sotl_errors, p_rs_sotl_latencies, p_true_errors,
                        p_true_latencies, rs_sotl_errors, pareto_front_rs_sotl[1],
                        archs_true_filtered, 'rs_sotl', metric, rays)
    update_metrics_dict(metrics_dict, p_rhpn_errors, p_rhpn_latencies, p_true_errors,
                        p_true_latencies, rhpn_errors_filtered, pareto_front_rhpn[1],
                        archs_true_filtered, 'rhpn', metric, rays)
    update_metrics_dict(metrics_dict, p_rhpn_sotl_errors, p_rhpn_sotl_latencies, p_true_errors,
                        p_true_latencies, rhpn_sotl_errors, pareto_front_rhpn_sotl[1],
                        archs_true_filtered, 'rhpn_sotl', metric, rays)

    # HELP NAS data
    if help_loader is not None:
        help_errors_filtered, help_latencies_filtered, help_archs_filtered, help_sotl_errors, help_sotl_latencies, help_sotl_archs = \
                get_sotl_data(data, model, archs_help, dataset, metric,
                              False, False, 12,
                              min_err, max_err, min_lat, max_lat)

        pareto_front_help = plot_pareto_frontier(help_errors_filtered,
                                                 help_latencies_filtered,
                                                 help_archs_filtered, maxX=False,
                                                 maxY=False, color="violet",
                                                 xlabel="Error", ylabel=metric,
                                                 marker='s', linestyle='-.',
                                                 legend="MetaD2A+HELP",
                                                 scatter=False)
        pareto_front_help_sotl = plot_pareto_frontier(help_sotl_errors,
                                                      help_sotl_latencies,
                                                      help_sotl_archs, maxX=False,
                                                      maxY=False, color="pink",
                                                      xlabel="Error", ylabel=metric,
                                                      marker='s', linestyle='-',
                                                      legend="MetaD2A+HELP (SoTL-EMA)",
                                                      scatter=True)

        p_help_errors, p_help_latencies = get_data_from_pareto(pareto_front_help[0])
        p_help_sotl_errors, p_help_sotl_latencies = get_data_from_pareto(pareto_front_help_sotl[0])
        update_metrics_dict(metrics_dict, p_help_errors, p_help_latencies, p_true_errors,
                            p_true_latencies, help_errors_filtered, pareto_front_help[1],
                            archs_true_filtered, 'help', metric, rays)
        update_metrics_dict(metrics_dict, p_help_sotl_errors, p_help_sotl_latencies, p_true_errors,
                            p_true_latencies, help_sotl_errors, pareto_front_help_sotl[1],
                            archs_true_filtered, 'help_sotl', metric, rays)

    save_name = "pareto" + "_" + str(epoch) + ".png"
    save_path = args.save + "/plots/{}".format(metric)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path)
    plt.clf()

    print("Metrics: ", metrics_dict)
    save_name = "metrics" + "_" + str(epoch) + ".pkl"
    save_path = args.save + "/statistics/{}".format(metric)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, save_name)
    with open(save_path, "wb") as f:
        pickle.dump(metrics_dict, f)

    return metrics_dict


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def save_args(folder, args, name="config.json", check_exists=False):
    set_logger()
    path = Path(folder)
    if check_exists:
        if path.exists():
            logging.warning(
                f"folder {folder} already exists! old files might be lost.")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(vars(args), open(path / name, "w"))


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def compute_ema_sotl(train_losses, epochs=12, mu=0.99):
    train_epochs = len(train_losses)
    EMA_SoTL = []
    for se in range(train_epochs):
        if se <= 0:
            ema = train_losses[se]
        else:
            ema = ema * (1 - mu) + mu * train_losses[se]
        EMA_SoTL.append(ema)
        score = np.sum(EMA_SoTL[:epochs])
    return score

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def process_step_vector(x, method, mask, tau=None):
    if method == 'softmax':
        output = F.softmax(x, dim=-1)
    elif method == 'dirichlet':
        output = torch.distributions.dirichlet.Dirichlet(
            F.elu(x) + 1).rsample()
    elif method == 'gumbel':
        output = F.gumbel_softmax(x, tau=tau, hard=False, dim=-1)

    if mask is None:
        return output
    else:
        output_pruned = torch.zeros_like(output)
        output_pruned[mask] = output[mask]
        output_pruned /= output_pruned.sum()
        assert (output_pruned[~mask] == 0.0).all()
        return output_pruned


def process_step_matrix(x, method, mask, tau=None):
    weights = []
    if mask is None:
        for line in x:
            weights.append(process_step_vector(line, method, None, tau))
    else:
        for i, line in enumerate(x):
            weights.append(process_step_vector(line, method, mask[i], tau))
    return torch.stack(weights)


def prune(x, num_keep, mask, reset=False):
    if not mask is None:
        x.data[~mask] -= 1000000
    src, index = x.topk(k=num_keep, dim=-1)
    if not reset:
        x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=src))
    else:
        x.data.copy_(torch.zeros_like(x).scatter(
            dim=1, index=index, src=1e-3*torch.randn_like(src)))
    mask = torch.zeros_like(x, dtype=torch.bool).scatter(
        dim=1, index=index, src=torch.ones_like(src, dtype=torch.bool))
    return mask


def compute_hypervolume(front, ref):
    hv = hypervolume(front)
    return hv.compute(ref)


def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def get_str_to_idx_dict(hw_api, dataset = "cifar10"):
    indices = list(range(15625))
    str_to_idx_map = {}
    for i in indices:
        config = hw_api.get_net_config(i, dataset)
        str_to_idx_map[config["arch_str"]] = i
    return str_to_idx_map

def add_global_node( mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return torch.FloatTensor(mx)

