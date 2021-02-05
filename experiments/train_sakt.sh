#!/bin/bash
export CUDA_VISIBLE_DEVICES="1,2,3"

strings=(
    "algebra05"
    "bridge_algebra06"
    "assistments09"
    "assistments12"
    "assistments15"
    "assistments17'"
    "ednet_small"
    "spanish"
    "statics"
)

for i in "${strings[@]}"; do
    python train_sakt.py "--setup=$i"
done
