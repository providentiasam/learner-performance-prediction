test_types=("insertion" "deletion" "replacement")
for tt in "${test_types[@]}"
do
        command="python behavior_test.py --dataset spanish --model saint --test_type ${tt} --load_dir ./save --filename spanish --diff_threshold 0 --gpu 5,6"
        echo "${command}"
        eval "${command}"
done
