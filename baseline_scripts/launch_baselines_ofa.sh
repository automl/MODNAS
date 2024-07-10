#!/bin/bash
methods=("NSGA2")
devices=("2080ti_32")

for method in "${methods[@]}"
do
    for device in "${devices[@]}"
    do
        exp_name="test-${method}-${device}"
        echo Submitting job $exp_name
        sbatch --bosch -J $exp_name baseline_scripts/baselines_ofa.sh $method $device
    done
done