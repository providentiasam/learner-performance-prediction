#!/bin/bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
datasets=("spanish" "statics" "ednet_small" "assistments15" "assistments17")
tests=("original" "insertion" "deletion" "replacement")
models=("dkt1" "saint")
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		for test in "${tests[@]}"; do
			python behavior_test.py --dataset=$dataset --model=$model --test_type=$test --filename=best
		done
	done
done
