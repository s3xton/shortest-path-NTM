cd ../
cd ../
for ((i=$1; i < $1 + 10; i++))
do
    python3 main.py --is_LSTM_mode True --is_test True --rand_hyper True --continue_train False --val_set_size 0 --checkpoint_dir "checkpoint_lstm/hyper/$i" --dataset_dir "dataset_files"
done