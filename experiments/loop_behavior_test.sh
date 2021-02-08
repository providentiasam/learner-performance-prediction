#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
datasets=("spanish" "statics" "ednet_small" "assistments15" "assistments17")
datasets=("ednet_small")
tests=("original" "insertion" "deletion" "replacement")
tests=("question_prior")
tests=("insertion" "deletion" "replacement")
models=("sakt" "dkt1" "saint")
models=("sakt")
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		for test in "${tests[@]}"; do
			python behavior_test.py --dataset=$dataset --model=$model --test_type=$test --filename=best
		done
	done
done
