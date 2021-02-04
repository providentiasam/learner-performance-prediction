indices=(0 1 2 3 4)
datasets=("spanish" "statics" "ednet_small" "assistments15" "assistments17")
train_batch=(100 50 100 100 100)
lr=(0.001 0.001 0.001 0.001 0.001)
warmup_step=(4000 4000 200 400 200)
dropout_rate=(0.2 0.2 0.1 0.2 0.2)
layer_count=(2 2 1 2 2)
head_count=(10 1 4 1 10)
dim_model=(50 200 256 50 200)
dim_ff=(200 800 1024 200 800)
seq_len=(200 200 100 100 100)
eval_steps=(5165 1761 568 5921 2623)

for i in "${indices[@]}"
do
    command="python train_saint.py --dataset=${datasets[i]} --train_batch=${train_batch[i]} --lr=${lr[i]} --warmup_step=${warmup_step[i]} --dropout_rate=${dropout_rate[i]} --layer_count=${layer_count[i]} --head_count=${head_count[i]} --dim_model=${dim_model[i]} --dim_ff=${dim_ff[i]} --seq_len=${seq_len[i]} --eval_steps=10000"
    echo "$command"
    eval "$command"
done