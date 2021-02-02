#!/bin/bash
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=$1
for warmup_step in 200 400
do
	for layer_count in 2 1
	do
		for dim_model in 200 100 50
		do
			for head_count in 10 5 1
			do
				python train_saint.py "--dataset=$2" "--warmup_step=$warmup_step" "--train_batch=100" "--layer_count=$layer_count" "--dim_model=$dim_model" "--dim_ff=$((4*dim_model))" "--head_count=$head_count" "--num_epochs=500" "--val_check_interval=0.5" "--gpu=0" "--seq_len=100" "--limit_train_batches=1.0"
			done
		done
	done
done

