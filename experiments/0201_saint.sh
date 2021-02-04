test_types=("replacement")
indices=(1 2 3 4)
dataset=("spanish" "statics" "ednet_small" "assistments15" "assistments17")
filename=("val_auc=0.8633" "val_auc=0.8282" "val_auc=0.7344" "val_auc=0.7201" "val_auc=0.7709")
for i in "${indices[@]}"
do
    for tt in "${test_types[@]}"
    do
        command="python behavior_test.py --dataset ${dataset[i]} --model saint --test_type ${tt} --load_dir ./save/${dataset[i]} --filename ${dataset[i]}/${filename[i]} --diff_threshold 0 --gpu 5,6"
        echo "${command}"
        eval "${command}"
    done
done

