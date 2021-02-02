#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

strings=(
    "ednet_small"
    "spanish"
)

for i in "${strings[@]}"; do
    python train_sakt.py "--setup=$i"
done
