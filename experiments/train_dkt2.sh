#!/bin/bash
datasets=("ednet_medium")
for dataset in "${datasets[@]}"; do
	for lr in 0.003 0.01 0.001
	do
		for num_hid_layers in 1 2
		do
			for hid_size in 50 100 200
			do
				for drop_prob in 0.25 0.5
				do
					python train_dkt2.py "--dataset=$dataset" "--lr=$lr" "--num_hid_layers=$num_hid_layers" "--hid_size=$hid_size" "--drop_prob=$drop_prob" "--num_epochs=50" "--gpu=0,1,2,3"
				done
			done
		done
	done
done

