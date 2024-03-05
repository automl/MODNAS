#!/bin/bash
#SBATCH -p <partition name>
#SBATCH --gres=gpu:8
#SBATCH -t 5-00:00:00 # time (D-HH:MM)
#SBATCH -c 32
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
export PYTHONPATH=.
python -m torch.distributed.launch --nproc_per_node=8 --use_env search_spaces/MobileNetV3/search/mobilenet_search_base.py --one_shot_opt reinmax --opt_strategy "simultaneous" --hpn_type meta --use_pretrained_hpn 
