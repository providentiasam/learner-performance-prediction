#!/bin/bash
seq_len=200
for warmup_step in 500 2000
do
	for layer_count in 4 3 2
	do
		for dim_model in 50 100 200
		do
			for head_count in 10 25 50
			do
				python train_saint.py "--dataset=ednet_medium" "--warmup_step=$warmup_step" "--train_batch=256" "--layer_count=$layer_count" "--dim_model=$dim_model" "--dim_ff=$((2*dim_model))" "--head_count=$head_count" "--num_epochs=20" "--val_check_interval=1" "--seq_len=${seq_len}" "--limit_train_batches=1.0"
			done
		done
	done
done

