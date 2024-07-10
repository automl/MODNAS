#!/bin/bash
methods=("NSGA2" "LSBO" "RSBO" "MOASHA" "EHVI")
devices=("cpu_xeon")

for method in "${methods[@]}"
do
    for device in "${devices[@]}"
    do
        exp_name="test-${method}-${device}"
        echo Submitting job $exp_name
        sbatch baseline_scripts/baselines_hat.sh $method $device
    done
done
