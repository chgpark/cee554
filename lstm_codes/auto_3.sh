#!/bin/sh
gpu='0'
batch=8977
num_uwb=3
train_dt="/home/shapelim/RONet/RO_train_3/"
val_dt="/home/shapelim/RONet/RO_val_3/"
test_dt="/home/shapelim/RONet/RO_test_3/02-38.csv"
####
seq=5
dir="/home/shapelim/RONet/0215_3_fc_"
network="fc"
count="_1/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 1 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_2/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.96 --decay_step 5 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_3/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 10 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_4/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 1 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_5/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.95 --decay_step 5 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_6/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch--decay_rate 0.99 --decay_step 20 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
#################
dir="/home/shapelim/RONet/0215_3_bi_"
network="stacked_bi"
count="_1/"
count="_1/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 1 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_2/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.96 --decay_step 5 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_3/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 10 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_4/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 1 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_5/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.95 --decay_step 5 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
count="_6/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch--decay_rate 0.99 --decay_step 20 --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network --train_data $train_dt --val_data $val_dt --num_uwb $num_uwb --test_data $test_dt
#
