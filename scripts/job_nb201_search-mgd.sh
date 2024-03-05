#!/bin/bash
#SBATCH -p <partition name>
#SBATCH --gres=gpu:8
#SBATCH -t 5-00:00:00 # time (D-HH:MM)
#SBATCH -c 32
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -a 9001-9003 # job array index

w_grad=$1
hpn_grad=$2
wd=$3
epochs=$4
seed=$SLURM_ARRAY_TASK_ID

lr_max=0.025
lr_min=0.001


source activate modnas
PYTHONPATH=. python search_spaces/nb201/search_nb201_mgd.py \
    --save mgd-100epochs \
    --wandb_name "modnas-nb201-100epochs" \
    --optimizer_type "reinmax" \
    --arch_weight_decay 0.09 \
    --train_portion 0.5 \
    --learning_rate $lr_max \
    --learning_rate_min $lr_min \
    --seed $seed \
    --epochs $epochs \
    --load_path "predictor_data_utils/nb201/predictor_meta_learned.pth" \
    --w_grad_update_method $w_grad \
    --hpn_grad_update_method $hpn_grad \
    --weight_decay $wd

