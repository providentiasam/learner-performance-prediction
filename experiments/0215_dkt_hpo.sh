for dataset in "ednet_small" "ednet_medium" "ednet"
do
    for hid_size in 50 100 200
    do
        for embed_size in 50 100 200
        do
            for num_hid_layers in 1 2
            do
                for drop_prob in 0 0.25 0.5
                do
                    for lr in 0.003
                    do
                        python train_dkt2.py "--dataset=$dataset" "--hid_size=$hid_size" "--embed_size=$embed_size" "--num_hid_layers=$num_hid_layers" "--drop_prob=$drop_prob" "--lr=$lr" "--num_epochs=300" "--project=bt_dkt2" "--name=${dataset}_hid${hid_size}_embed${embed_size}_layer${num_hid_layers}_drop${drop_prob}_lr${lr}"
                    done
                done
            done
        done
    done
done