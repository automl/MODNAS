####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################

import os
import torch
from optimizers.help.parser import get_parser
from optimizers.help.help_ofa import HELP

def run_nas(args):
    set_seed(args)
    args.gpu = int(args.gpu)
    args = set_path(args)

    print(f'==> mode is [{args.mode}] ...')
    model = HELP(args)
    acc, lat, arch = model.nas()
    return acc, lat, arch

def set_seed(args):
    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_path(args):
    args.data_path = os.path.join(
        args.main_path, 'datasets/help', args.search_space)
    args.save_path = os.path.join(
        "experiments_mgd", args.exp_name)
    #args.save_path = os.path.join(
            #args.save_path, args.search_space)
    #args.save_path = os.path.join(args.save_path, args.exp_name)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    #print(f'==> save path is [{args.save_path}] ...')
    return args

if __name__ == '__main__':
    run_nas(get_parser())
