test_types=("original" "insertion" "deletion" "replacement")
indices=(0)
dataset=("ednet_medium")
filename=("val_auc=0.7656")
for i in "${indices[@]}"
do
    for tt in "${test_types[@]}"
    do
        command="python behavior_test.py --dataset ${dataset[i]} --model saint --test_type ${tt} --load_dir ./save/${dataset[i]} --filename ${filename[i]} --diff_threshold 0 --gpu 5,6"
        echo "${command}"
        eval "${command}"
    done
done

