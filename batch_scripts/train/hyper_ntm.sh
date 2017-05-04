for ((i=1; i <= 5; i++))
do
    python3 /home/sexton/Documents/Dissertation/main.py --is_LSTM_mode False --is_train True --rand_hyper True --continue_train False --train_set_size 40000 --checkpoint_dir "/home/conor/Documents/Dissertation/checkpoint_ntm/hyper/$i" --dataset_dir "/home/conor/Documents/Dissertation/dataset_files"
done