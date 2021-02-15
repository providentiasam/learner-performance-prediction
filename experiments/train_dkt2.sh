#!/bin/bash
export CUDA_VISIBLE_DEVICES="4,5"
datasets=("ednet_small" "ednet_medium" "ednet" "assistments15" "assistments17" "spanish" "statics" )
for dataset in "${datasets[@]}"; do
	for lr in $(seq 0.001 0.003 0.01)
	do
		for num_hid_layers in 1 2
		do
			for hid_size in 50 100 200
			do
				for drop_prob in $(seq 0.25 0.5)
				do
					python train_dkt2.py "--dataset=$dataset" "--lr=$lr" "--num_hid_layers=$num_hid_layers" "--hid_size=$hid_size" "--drop_prob=$drop_prob" "--num_epochs=50"
				done
			done
		done
	done
done

