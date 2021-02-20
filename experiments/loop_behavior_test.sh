#!/bin/bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
datasets=("spanish" "statics" "assistments15" "assistments17" "ednet_small" "ednet_medium" "ednet")

datasets=("spanish" "statics" "assistments15" "assistments17" "ednet_small")

tests=("original" "insertion" "deletion" "question_prior" "replacement" "repetition" "continuity")
models=("dkt" "saint" "sakt_legacy" "sakt")
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		for test in "${tests[@]}"; do
			python behavior_test.py --dataset=$dataset --model=$model --test_type=$test --filename=best
		done
	done
done

datasets=("ednet_medium" "ednet")
for model in "${models[@]}"; do
	for dataset in "${datasets[@]}"; do
		for test in "${tests[@]}"; do
			python behavior_test.py --dataset=$dataset --model=$model --test_type=$test --filename=best
		done
	done
done