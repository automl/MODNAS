#!/bin/bash
methods=("NSGA2" "EHVI")
devices=("eyeriss" "fpga")
seeds=(9001 9002 9003 9004)
for method in "${methods[@]}"
do
    for device in "${devices[@]}"
    do
        for seed in "${seeds[@]}"
        do
            exp_name="test-${method}-${device}-${seed}"
            echo Submitting job $exp_name
            sbatch --bosch -J $exp_name baseline_scripts/baselines_nb201_3_obj.sh $method $device $seed
        done
    done
done