configs=$1
metrics=${2:-"sacre"}
gpu=${3:-0}
subset=${4:-"test"}
scalarization=${5:-0}
device=${6:-"cpu_xeon"}
task=${7:-"wmt14.en-de"}
data_path=${8:-"data/binary/wmt16_en_de"}
model_path=${9:-"/path/to/model.pt"}
output_path=$(dirname -- "results2")
out_name=$(basename -- "$configs")
echo "$output_path"
mkdir -p results_final/exp

CUDA_VISIBLE_DEVICES=$gpu python search_spaces/hat/generate.py --task_type $task --scalarization $scalarization --data $data_path   --path $model_path --gen-subset test --beam 4 --batch-size 128 --remove-bpe  --lenpen 0.6 --configs $configs --device $device > results_final/exp/${out_name}_${subset}_${scalarization}_${device}_${task}_gen.out