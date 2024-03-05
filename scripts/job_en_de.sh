#!/bin/bash
#SBATCH -p <partition name>
#SBATCH --gres=gpu:8
#SBATCH -t 5-00:00:00 # time (D-HH:MM)
#SBATCH -c 32
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
python search_spaces/hat/train.py --configs=search_spaces/hat/configs/wmt14.en-de/supertransformer/space0.yml
