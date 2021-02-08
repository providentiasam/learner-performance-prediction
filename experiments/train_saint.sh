#!/bin/bash
export PYTHONPATH=$PWD
#export CUDA_VISIBLE_DEVICES=$1
seq_len=100
for warmup_step in 4000
do
	for layer_count in 2
	do
		for dim_model in 200
		do
			for head_count in 10
			do
				python train_saint.py "--dataset=ednet_medium" "--warmup_step=$warmup_step" "--train_batch=100" "--layer_count=$layer_count" "--dim_model=$dim_model" "--dim_ff=$((4*dim_model))" "--head_count=$head_count" "--num_epochs=500" "--val_check_interval=0.5" "--gpu=3" "--seq_len=${seq_len}" "--limit_train_batches=1.0"
			done
		done
	done
done

