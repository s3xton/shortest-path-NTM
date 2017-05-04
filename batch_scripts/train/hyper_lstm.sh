cd ../
cd ../
for ((i=$1; i <= $1 + 5; i++))
do
    python3 main.py --is_LSTM_mode True --is_train True --rand_hyper True --continue_train False --train_set_size 40000 --checkpoint_dir "checkpoint_lstm/hyper/$i" --dataset_dir "dataset_files"
done