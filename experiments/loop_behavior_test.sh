#!/bin/bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
datasets=("spanish" "statics" "ednet_small" "assistments15" "assistments17" "ednet_medium")
datasets=("ednet_medium")
tests=("insertion" "deletion" "question_prior" "replacement" "original")
models=("saint")
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		for test in "${tests[@]}"; do
			python behavior_test.py --dataset=$dataset --model=$model --test_type=$test --filename=best
		done
	done
done
