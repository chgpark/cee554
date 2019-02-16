##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_18/" --alpha 0.1 --gamma 0.1
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_18/" --alpha 0.1 --gamma 0.1
##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_19/" --alpha 0.1 --gamma 0.3
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_19/" --alpha 0.1 --gamma 0.3
##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_20/" --alpha 0.1 --gamma 0.5
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_20/" --alpha 0.3 --gamma 0.5
##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_21/" --alpha 0.01 --gamma 0.1
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_21/" --alpha 0.01 --gamma 0.1
##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_22/" --alpha 0 --gamma 0
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_22/" --alpha 0 --gamma 0
##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_23/" --alpha 0.01 --gamma 0.1
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_23/" --alpha 0.3 --gamma 0.3
##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_24/" --alpha 0.001 --gamma 1
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_24/" --alpha 0.001 --gamma 1
##############
python3 train_cudnn_with_loss.py --save_dir "/home/shapelim/RONet/loss_term_25/" --alpha 0.001 --gamma 0.4
python3 test_cudnn_with_loss.py --load_model_dir "/home/shapelim/RONet/loss_term_25/" --alpha 0.0001 --gamma 0.4


