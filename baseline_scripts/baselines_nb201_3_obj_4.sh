#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #rtx2080
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH --mem=100G
#SBATCH -o logs_final/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs_final/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)

method=$1
device=$2
seed=$3

python baselines/run_baselines_nb201_3_obj.py --method $method --metric_latency $device --metric_energy $device --random_seed $seed