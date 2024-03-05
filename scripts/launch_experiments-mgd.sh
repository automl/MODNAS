#!/bin/bash
w_grad=("mean")
hpn_grad=("mgd")
hpn_type=("meta")
wd_w=(0.0027)
sample_per_device=(1)
lr_type=(1)
epochs=(100)
cosine_sim_sch=(0)

for hpn in "${hpn_type[@]}"
do
    for h in "${hpn_grad[@]}"
    do
	for w in "${w_grad[@]}"
        do
		for p in "${arch_lr_sch[@]}"
		do
			for l in "${wd_w[@]}"
			do
				for lr in "${lr_type[@]}"
				do
					for s in "${sample_per_device[@]}"
					do
						for o in "${epochs[@]}"
						do
                            for sw in "${cosine_sim_sch[@]}"
                            do
                                        exp_name="${h}-${w}-${hpn}-${p}-${l}-${lr}-${s}-${o}-${sw}"
                                        echo Submitting job $exp_name
                                
                                        sbatch -J $exp_name scripts/job_nb201_search-mgd.sh $w $h $p $o
                            done
						done
					done
				done
			done
		done
        done
    done 
done
