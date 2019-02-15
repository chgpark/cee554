#!/bin/sh
gpu='0'
seq=8
dir="/home/shapelim/RONet/0215_8_fc_"
network="fc"
batch=8972
#
#
count="_1/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 1 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_3/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.96 --decay_step 5 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_5/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 10 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_2/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.96 --decay_step 1 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_4/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.9 --decay_step 5 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_6/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch--decay_rate 0.99 --decay_step 20 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#################
seq='5'
dir="/home/shapelim/RONet/0215_8_bi_"
network="stacked_bi"
count="_1/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 1 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_3/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.96 --decay_step 5 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_5/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.98 --decay_step 10 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_2/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.96 --decay_step 1 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_4/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch --decay_rate 0.9 --decay_step 5 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
#
count="_6/"
python3.5 train.py --save_dir $dir$count --gpu $gpu --sequence_length $seq --batch_size $batch--decay_rate 0.99 --decay_step 20 --network_type $network
python3.5 test.py --load_model_dir $dir$count --gpu $gpu --sequence_length $seq --network_type $network
