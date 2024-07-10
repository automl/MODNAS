#!/bin/bash
methods=("NSGA2" "EHVI")
devices=("1080ti_1" "1080ti_32" "1080ti_256" "silver_4114" "titan_rtx_256" "gold_6226" "silver_4210r" "samsung_a50" "pixel3" "essential_ph_1" "fpga" "pixel2"
 "samsung_s7" "titanx_1" "titanx_32" "titanx_256" "gold_6240" "raspi4" "eyeriss")
seeds=(9001 9002 9003 9004)
for method in "${methods[@]}"
do
    for device in "${devices[@]}"
    do
        for seed in "${seeds[@]}"
        do
            exp_name="test-${method}-${device}-${seed}"
            echo Submitting job $exp_name
            sbatch --bosch -J $exp_name baseline_scripts/baselines_nb201.sh $method $device $seed
        done
    done
done