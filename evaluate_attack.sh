#!/bin/bash
if [ $# -lt 3 ]
then
	echo "Usage: $(basename $0) output_dir labels_file graph_file"
	exit 1
fi
output_dir=$1
result_dir="$output_dir/result"
labels_file=$2
graph_file=$3

python3 evaluate_attack.py --output_dir=$result_dir --labels_file=$labels_file --graph_file=$graph_file
