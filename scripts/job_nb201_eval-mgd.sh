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
hpn=$3
arch_lr_schedule=$4
wd=$5
lr_type=$6
sample_per_device=$7
epochs=$8
cosine_schedule=$9
seed=$SLURM_ARRAY_TASK_ID



if [ $arch_lr_schedule -eq 1 ]; then
    arch_lr_sch="--use_arch_lr_scheduler"
else
    arch_lr_sch=""
fi

if [ $lr_type -eq 1 ]; then
    lr_max=0.025
    lr_min=0.001
else
    lr_max=0.01
    lr_min=0.0006
fi

if [ $cosine_schedule -eq 1 ]; then
    cosine_sch="--use_cosine_sim_schedule"
else
    cosine_sch=""
fi

if [ $sample_per_device -eq 1 ]; then
    sc_per_device="--sample_scalarization_per_device"
else
    sc_per_device=""
fi

source activate modnas
PYTHONPATH=. python search/test_nb201_mgd.py \
    --save mgd-batch-stats \
    --optimizer_type "reinmax" \
    --arch_weight_decay 0.09 \
    --train_portion 0.5 \
    --learning_rate $lr_max \
    --learning_rate_min $lr_min \
    --seed $seed \
    --entangle_weights \
    --use_pretrained_hpn \
    --epochs $epochs \
    --hpn_type $hpn \
    --hw_embed_on \
    --load_path "predictor_data_utils/nb201/predictor_meta_learned.pth" \
    --num_random_devices 50 \
    --w_grad_update_method $w_grad \
    --hpn_grad_update_method $hpn_grad \
    $sc_per_device \
    --weight_decay $wd \
    --subtract_cosine_sim \
    $cosine_sch \
    $arch_lr_sch

