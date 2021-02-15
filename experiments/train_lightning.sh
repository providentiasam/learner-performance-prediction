#!/bin/bash
datasets=("ednet_small" "assistments15" "assistments17" "spanish" "statics" )
models=("sakt" "saint")
for seq_len in 200 400
do 
	for dataset in "${datasets[@]}"
	do
		for layer_count in 2 1 3
		do
			for dim_model in 100 200
			do
				for head_count in 10 25
				do
					for model in "${models[@]}"
					do
						command="python train_lightning.py --model=$model --dataset=$dataset \
						--train_batch=256 --layer_count=$layer_count --dim_model=$dim_model \
						 --dim_ff=$((4*dim_model)) --head_count=$head_count --num_epochs=50 \
						  --val_check_interval=1 --seq_len=${seq_len} --limit_train_batches=1.0"
						eval "$command"
					done
				done
			done
		done
	done
done

