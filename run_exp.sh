#!/usr/bin/env bash


# Probing classifier: 1-nn
BASE_DIR=$(pwd) # path to the project directory


# The pos and neg pairs are sampled via the path-based method
# Detailed results of each of the 3 runs are saved
train_path=$BASE_DIR/data/fullprobe/full-train
test_path=$BASE_DIR/data/fullprobe/full-test/test-3996

cd $train_path

arr1=(concat)
for d in */; do
	echo "$d"
	for i in "${arr1[@]}"; do
		python $BASE_DIR/knn_probe.py -train_dir $d -test_dir $test_path --input_mode $i --probing_mode full
	done
done

# Perform analysis
python $BASE_DIR/more_analysis.py -dir $test_path --probing_mode full




