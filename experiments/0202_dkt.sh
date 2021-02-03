test_types=("insertion" "deletion" "replacement")
weight_files=('spanish,batch_size=100,hid_size=100,num_hid_layers=1,drop_prob=0.25,lr=0.001' 'spanish,batch_size=50,hid_size=100,num_hid_layers=2,drop_prob=0.5,lr=0.001' 'spanish,batch_size=100,hid_size=50,num_hid_layers=1,drop_prob=0.25,lr=0.001' 'spanish,batch_size=50,hid_size=200,num_hid_layers=2,drop_prob=0.25,lr=0.001' 'spanish,batch_size=25,hid_size=50,num_hid_layers=2,drop_prob=0.5,lr=0.01'
)
for tt in "${test_types[@]}"
do
    for wf in "${weight_files[@]}"
    do
        command="python behavior_test.py --dataset spanish --model dkt1 --test_type ${tt} --load_dir ./save --filename ${wf} --diff_threshold 0 --gpu 5,6"
        echo "${command}"
        eval "${command}"
        break
    done
done
