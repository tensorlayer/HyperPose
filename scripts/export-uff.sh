#!/bin/sh
#this script should be called from the root dir
export_uff()
{
    local model_type=$1
    local model_name=$2
    local output_dir=$3
    source activate tf2
    echo "converting ${model_name} into pb file..."
    python ../export_pb.py --model_type=${model_type} --model_name=${model_name} --output_dir=${output_dir}
    conda deactivte
    source activate tf1
    echo "converting ${model_name} into uff file..."
    convert-to-uff -o ${output_dir}/${model_name}.uff ${output_dir}/frozen_${model_name}.uff
    echo "convertion finished!"
}

export_uff $1 $2 $3