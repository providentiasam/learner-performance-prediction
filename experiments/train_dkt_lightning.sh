indices=(3 4)
datasets=("spanish" "statics" "ednet_small" "assistments15" "assistments17")
hid_size=(50 200 100 50 200)
num_hid_layers=(2 1 1 1 2)
drop_prob=(0.5 0.5 0.5 0.5 0.5)
batch_size=(100 100 100 100 50)
lr=(0.01 0.01 0.001 0.001 0.01)

for i in "${indices[@]}"
do
    command1="python train_lightning.py --dataset=${datasets[i]} --model=dkt\
     --lr=${lr[i]} --layer_count=${num_hid_layers[i]} --dim_model=${hid_size[i]} \
     --dropout_rate=${drop_prob[i]} --train_batch=${batch_size[i]} --num_epochs=100 \
     --seq_len=10000 --stride=1000"
    echo $command1
    eval $command1
done