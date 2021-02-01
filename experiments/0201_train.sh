indices=(0 1 2 3 4)
hid_size=(50 100 100 50 200)
num_hid_layers=(2 1 2 1 2)
drop_prob=(0.5 0.25 0.5 0.25 0.25)
batch_size=(100 100 50 100 50)
lr=(0.01 0.001 0.001 0.001 0.001)
num_epochs=(100 200 120 300 120)

for i in "${indices[@]}"
do
    command1="python train_dkt1.py --dataset=spanish --lr=${lr[i]} --num_hid_layers=${num_hid_layers[i]} --hid_size=${hid_size[i]} --drop_prob=${drop_prob[i]} --batch_size=${batch_size[i]} --num_epochs=${num_epochs[i]}"
    echo $command1
    eval $command1
done