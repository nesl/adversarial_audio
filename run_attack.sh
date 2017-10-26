#!/bin/bash

if [ $# -lt 5 ]
    then
    echo "Usage: $(basename $0) dataset_dir ckpts_dir limit max_iters test_size."
    exit 1
fi
dataset_dir=$1
ckpts_dir=$2
limit=$3
max_iters=$4
test_size=$5


frozen_graph="$ckpts_dir/conv_actions_frozen.pb"
labels_file="$ckpts_dir/conv_actions_labels.txt"

if [ ! -d $ckpts_dir ]; then
    echo "Checkpoints dir does not exist."
    exit 1
fi

if [ ! -f $frozen_graph ]; then
    echo "Frozen graph does not exist."
    exit 1
fi

if [ ! -f $labels_file ]; then
    echo "Labels file does not exist."
    exit 1
fi
labels_list=`cat $labels_file`
sample_size=100


# remove output dir if exists
[ -d output ] && rm -rf "output/"
mkdir output

# Copy data files to output dir
for label in `ls $dataset_dir`
do
#TODO(malzantot) : see why this line ignores the labels with __
    if [ -d "$dataset_dir/$label" ] && [[ $labels_list == *"$label"* ]]; then
        mkdir -p "output/data/$label"
        find "$dataset_dir/$label/" -name "*.wav" | sort -R \
        | head -n$test_size | xargs -L1 cp -t "output/data/$label"
    fi
done

mkdir -p "output/result/"
for target_label in `ls output/data/`
do
    for source_label in `ls output/data`
    do
        if [ $source_label == $target_label ]; then
            continue
        fi
        echo "Running attack: $source_label --> $target_label"
        output_dir="output/result/$target_label/$source_label"
        mkdir -p $output_dir
        python audio_attack.py \
        --data_dir="output/data/$source_label" \
        --output_dir=$output_dir \
        --target_label=$target_label \
        --labels_path=$labels_file \
        --graph_path=$frozen_graph \
        --limit=$limit \
        --max_iters=$max_iters
    done
done
