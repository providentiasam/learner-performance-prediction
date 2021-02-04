test_types=("original" "insertion" "deletion" "replacement")
indices=(0 1 3)
datasets=("spanish" "statics" "ednet_small" "assistments15" "assistments17")
weight_files=('spanish,batch_size=50,hid_size=50,num_hid_layers=2,drop_prob=0.5,lr=0.01' 'statics,batch_size=50,hid_size=200,num_hid_layers=1,drop_prob=0.5,lr=0.01' '' 'assistments15,batch_size=100,hid_size=50,num_hid_layers=1,drop_prob=0.0,lr=0.001' ''
)

for i in "${indices[@]}"
do
    for tt in "${test_types[@]}"
    do
        command="python behavior_test.py --dataset ${datasets[i]} --model dkt1 --test_type ${tt} --load_dir ./save --filename ${weight_files[i]} --diff_threshold 0 --gpu 5,6"
        echo "${command}"
        eval "${command}"
    done
done

