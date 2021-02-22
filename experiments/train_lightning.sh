#!/bin/bash
datasets=("ednet_medium" "ednet" "assistments15" "assistments17" "spanish" "statics")
models=("sakt")
for dataset in "${datasets[@]}"
do 
	for seq_len in 200 500 1000 10000
	do
		for layer_count in 1 2 4
		do
			for dim_model in 100 200
			do
				for head_count in 10
				do
					for model in "${models[@]}"
					do
						command="python train_lightning.py --model=$model --dataset=$dataset \
						--train_batch=1024 --layer_count=$layer_count --dim_model=$dim_model \
						 --dim_ff=$((2*dim_model)) --head_count=$head_count --num_epochs=100 \
						  --val_check_interval=1 --seq_len=${seq_len} --stride=$((seq_len/2))\
						  --optimizer='noam' --lr=0.003 --gpu='0,1,2,3'"
						eval "$command"
					done
				done
			done
		done
	done
done

