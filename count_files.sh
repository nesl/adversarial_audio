#!/bin/bash
exp_data_dir="output/data"
exp_result_dir="output/result"
count_data=`find $exp_data_dir -name *.wav | wc -l`
count_result=`find $exp_result_dir -name *.wav | wc -l`
echo "Number of input files = $count_data."
echo "Number of output files = $count_result."
