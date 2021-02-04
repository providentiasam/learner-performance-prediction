indices=(4)
datasets=("spanish" "statics" "ednet_small" "assistments15" "assistments17")
hid_size=(50 200 100 50 200)
num_hid_layers=(2 1 1 1 2)
drop_prob=(0.5 0.5 0.5 0 0.5)
batch_size=(100 100 6 100 50)
lr=(0.01 0.01 0.001 0.001 0.01)
num_epochs=(92 63 7 24 8)

for i in "${indices[@]}"
do
    command1="python train_dkt1.py --dataset=${datasets[i]} --lr=${lr[i]} --num_hid_layers=${num_hid_layers[i]} --hid_size=${hid_size[i]} --drop_prob=${drop_prob[i]} --batch_size=${batch_size[i]} --num_epochs=20000"
    echo $command1
    eval $command1
done