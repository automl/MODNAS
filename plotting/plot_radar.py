from search.utils import circle_points, get_meta_d2a_data, get_ensemble_data
from search_spaces.nb201.model_search import NASBench201SearchSpace
from predictors.help.loader import Data
from plots.radar_plot import ComplexRadar

import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib
import pickle
import argparse
import numpy as np
import glob
import os
import re
import sys

matplotlib.use("Agg")

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = 'dotted'
plt.rcParams['font.size'] = 14

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='datapath',
                    help='location of the data corpus')
parser.add_argument('--run_help', action='store_true',
                    default=False, help='report frequency')
parser.add_argument('--run_ensemble', action='store_true',
                    default=False, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250,
                    help='num of training epochs')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()


pattern_str="mgd-batch-stats-search-*-reinmax-cifar10-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False"

def find_matching_dir_from_args(directory, pattern_str):
    matching_dirs = []
    escaped_pattern = re.escape(pattern_str)
    regex_pattern = re.compile(escaped_pattern.replace(r'\*', r'(.*?)'))

    for d in os.listdir(directory):
        if regex_pattern.search(d):
            matching_dirs.append(os.path.join(directory, d))

    return matching_dirs

hpn_load_paths = find_matching_dir_from_args("experiments/experiments_mgd/nasbench201", pattern_str)
print(hpn_load_paths)

#hpn_load_paths = \
#['experiments/experiments_mgd/nasbench201/mgd-batch-stats-search-20240115-002923-9003-reinmax-cifar10-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False',
# 'experiments/experiments_mgd/nasbench201/mgd-batch-stats-search-20240114-150323-9002-reinmax-cifar10-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False']

hpn_load_paths = \
["experiments_mgd_final/mgd-100epochs/search-20240118-114344-9007-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0",
"experiments_mgd_final/mgd-100epochs/search-20240118-114344-9008-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0",
"experiments_mgd_final/mgd-100epochs/search-20240118-114344-9009-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0",
"experiments_mgd_final/mgd-100epochs/search-20240118-114344-9010-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0",
"experiments_mgd_final/mgd-100epochs/search-20240118-114345-9003-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0",
"experiments_mgd_final/mgd-100epochs/search-20240120-172857-9002-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0-False",
"experiments_mgd_final/mgd-100epochs/search-20240120-172857-9005-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0-False",
"experiments_mgd_final/mgd-100epochs/search-20240120-172857-9006-reinmax-0.0027-False-100-meta-0.5-50-10-1-True-mean-mgd-True-True-False-0.09-0.025-0.001-False-False-False-101-1.0-False-0.0-False"
]

def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True

    if args.run_help or args.run_ensemble:
        model = NASBench201SearchSpace(16, 5, 4, 10,
                                       metric="fpga_latency", #placeholder, not used
                                       use_we_v2=False,
                                       entangle_weights=True,
                                       optimizer_type="reinmax",
                                       latency_norm_scheme="predictor_stats",
                                       hw_embed_on=True,
                                       hw_embed_dim=10,
                                       layer_size=100,
                                       load_path="predictor_data_utils/nb201/predictor_meta_learned.pth")
        model = model.cuda()
        model.eval()
    else:
        model = None

    help_loader = Data(mode='nas',
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

    metrics_moo = ['hypervolume', 'gd', 'gd_plus', 'igd', 'igd_plus']
    _methods = [
        'predicted', 'hpn_ensemble', 'help', 'rs', 'rhpn',
        #'sotl', 'rs_sotl', 'rhpn_sotl'
    ]
    ranges = {
        'gd': (0.0, 0.1),
        'gd_plus': (0.0, 0.1),
        'igd': (0.0, 0.17),
        'igd_plus': (0.0, 0.11),
        'hypervolume': (0.67, 1.0)
    }
    titles = {
        'gd': "GD",
        'gd_plus': "GD+",
        'igd': "IGD",
        'igd_plus': "IGD+",
        'hypervolume': "Hypervolume"
    }

    DEVICES_TRAIN = \
        help_loader.meta_train_devices+help_loader.meta_valid_devices+help_loader.meta_test_devices
    device_dictionary = load_modnas_results_devices(
        hpn_load_paths,
        DEVICES_TRAIN,
        model,
        99,
        help_loader,
        args.run_help,
        args.run_ensemble
    )
    #for m in metrics_moo:
        #r = ranges[m]
        #if m != "hypervolume":
            #methods = ['predicted', 'help', 'rs', 'rhpn']
        #else:
            #methods = ['predicted', 'help', 'rs', 'rhpn', 'true']
        #plot_radar(f'radar_{m}_train.pdf', device_dictionary, m, methods,
                   #r)

    DEVICES_TEST = help_loader.meta_test_devices
    #device_dictionary = load_modnas_results_devices(
        #hpn_load_paths,
        #DEVICES_TEST,
        #model,
        #99,
        #help_loader,
        #args.run_help
    #)
    for m in metrics_moo:
        r = ranges[m]
        if m == "hypervolume":
            methods = _methods + ['true']
        else:
            methods = _methods
        plot_radar(f'radar_{m}.pdf', device_dictionary, DEVICES_TEST, m, methods, r)


def load_modnas_results_devices(hpn_load_path, devices, model, epoch, help_loader,
                                run_help, run_ensemble):
    device_d = dict()
    for d in devices:
        print('DEVICE: %s'%d)
        d_mean, d_std = load_modnas_results(hpn_load_paths, d,
                                            model,
                                            epoch,
                                            help_loader,
                                            run_help=run_help,
                                            run_ensemble=run_ensemble)
        print(d_mean)
        device_d[d] = (d_mean, d_std)
    return device_d

def load_modnas_results(paths, device, model, epoch, help_loader,
                        run_help=False, run_ensemble=False):
    dict_list = list()
    circle_points_test = circle_points(24)
    for seed, p in enumerate(paths):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        file = os.path.join(
            p, "statistics", device+'_latency', f"metrics_{epoch}.pkl"
        )
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if run_help:
            print('>>>> Running HELP...')
            help_dict = get_meta_d2a_data(model,
                                          device,
                                          epoch,
                                          p,
                                          "cifar10",
                                          seed=seed,
                                          rays=np.array(circle_points_test),
                                          help_loader=help_loader)
            data.update(help_dict)
        else:
            file_help = os.path.join(
                p, "statistics", device+'_latency', "metrics_help.pkl"
            )
            if os.path.exists(file_help):
                with open(file_help, 'rb') as f:
                    help_dict = pickle.load(f)
                data.update(help_dict)


        dict_list.append(data)

    mean_dict, std_dict = dict(), dict()
    for k in dict_list[0].keys():
        list_of_values = [x[k] for x in dict_list]
        mean_dict.update({k: np.mean(list_of_values)})
        std_dict.update({k: 1.96*np.std(list_of_values)/np.sqrt(len(list_of_values))})

    if run_ensemble:
        print('>>>> Creating Ensemble predictions.')
        p = 'experiments/experiments_mgd/nasbench201/'
        ens_dict = get_ensemble_data(paths,
                                     model,
                                     device,
                                     epoch,
                                     p,
                                     'cifar10',
                                     seed=seed,
                                     rays=np.array(circle_points_test),
                                     help_loader=help_loader)
    else:
        p = 'experiments_mgd_final/mgd-100epochs'
        file_ensemble = os.path.join(
            p, "ensemble", device+'_latency', 'metrics_ensemble.pkl'
        )
        if os.path.exists(file_ensemble):
            with open(file_ensemble, 'rb') as f:
                ens_dict = pickle.load(f)
        else:
            print(f'>>>> Ensemble path {file_ensemble} not found...')
            ens_dict = {}
    mean_dict.update(ens_dict)

    return mean_dict, std_dict

def plot_radar(savename, device_dict, devices_test, metric='hypervolume',
               methods=['predicted', 'help', 'rs', 'rhpn', 'true'],
               ranges=(0.0, 1.0)):
    # Radar plot
    fig = plt.figure(figsize=(7, 7))
    categories = list(device_dict.keys())
    ranges = [ranges]*len(categories)
    N = len(methods)

    colors = {
        'predicted': "crimson",
        'help': "darkorange",
        'rs': "dodgerblue",
        'rhpn': "forestgreen",
        'true': "black",
        'sotl': "magenta",
        'hpn_ensemble': "cyan",
    }
    markers = {
        'predicted': "v",
        'help': "h",
        'rs': "x",
        'rhpn': "^",
        'true': ".",
        'sotl': ">",
        'hpn_ensemble': "<",
    }
    lstyle = {
        'predicted': "solid",
        'help': "solid",
        'rs': "solid",
        'rhpn': "solid",
        'true': "-.",
        'sotl': "solid",
        'hpn_ensemble': "--",
    }
    labels = {
        'predicted': "MODNAS",
        'help': "HELP",
        'rs': "RS",
        'rhpn': "RHPN",
        'true': "True",
        'sotl': "SoTL",
        'hpn_ensemble': "MODNAS-E (2)",
    }

    radar = ComplexRadar(fig, categories, ranges, metric, devices_test,
                         label_fontsize=11)
    for i, m in enumerate(methods):
        values = np.array([device_dict[d][0][f'{d}_latency_{metric}_{m}'] for d in
                          categories])
        #std_err = np.array([device_dict[d][1][f'{d}_latency_{metric}_{m}'] for d in
                          #categories])
        radar.plot(values, annotate=False, linewidth=2, color=colors[m],
                   linestyle=lstyle[m], marker=markers[m], markersize=8,
                   label=labels[m])
        radar.fill(values, color=colors[m], alpha=.05)
        #radar.errorbar(values, std_err, color=colors[m], alpha=.05)

    if metric == "gd_plus":
        radar.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                        fancybox=True, shadow=False, ncol=3)
    else:
        l = radar.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                            fancybox=True, shadow=False, ncol=3)
        l.set_visible(False)
    plt.grid(color='#AAAAAA')
    radar.ax.set_facecolor('#FAFAFA')
    radar.ax.spines['polar'].set_color('#222222')

    titles = {
        'gd': "GD",
        'gd_plus': "GD+",
        'igd': "IGD",
        'igd_plus': "IGD+",
        'hypervolume': "Hypervolume"
    }
    #plt.title(titles[metric])

    os.makedirs('plots/radar_plots', exist_ok=True)
    plt.savefig(f'plots/radar_plots/{savename}',
                bbox_inches="tight",
                #pad_inches=0.09,
                dpi=100,
               )
    plt.close("all")

if __name__ == '__main__':
    d = main()


