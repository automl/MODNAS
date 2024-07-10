#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1
#SBATCH -o logs_final/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs_final/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
method=$1
device=$2
python baselines/run_baselines_ofa.py --method $method --metric $device