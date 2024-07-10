#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080 #rtx2080
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:8
#SBATCH --mem=100G
#SBATCH -o logs_final/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs_final/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
method=$1
device=$2
python baselines/run_baselines_hat.py --configs=configs/wmt14.en-de/supertransformer/space0.yml --arch transformersuper_wmt_en_de  --max-tokens 4096 --encoder-embed-dim 640 --decoder-embed-dim 640 --qkv-dim 512 --encoder-ffn-embed-dim 3072 --decoder-ffn-embed-dim 3072 --metric $device --method $method