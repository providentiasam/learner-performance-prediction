#!/bin/bash
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=$1
for seq_len in 100
do
	for layer_count in 1 
	do
		for dim_model in 50
		do
			for head_count in 5
			do
				python train_saint.py "--dataset=$2" "--layer_count=$layer_count" "--dim_model=$dim_model" "--head_count=$head_count" "--num_epochs=100" "--val_check_interval=0.5" "--gpu=0" "--seq_len=$seq_len" "--limit_train_batches=1.0"
			done
		done
	done
done

