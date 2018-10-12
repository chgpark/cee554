# Lab 12 Character Sequence RNN
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from lstm_network import RONet
import numpy as np
import DataPreprocessing
from tqdm import tqdm, trange
import os
import argparse
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.set_random_seed(777)  # reproducibilityb
# hyper parameters
p =argparse.ArgumentParser()
#FOR TRAIN
p.add_argument('--train_data', type=str, default="train_data.csv")
p.add_argument('--board_dir', type=str, default="./board/LSTM/stacked_bi_epoch_1000_lr_0_02")
p.add_argument('--save_dir', type=str, default="model/LSTM/stacked_bi_epoch_1000_lr_0_02/")

p.add_argument('--lr', type=float, default = 0.014)
p.add_argument('--decay_rate', type=float, default = 0.7)
p.add_argument('--decay_step', type=int, default = 7)
p.add_argument('--epoches', type=int, default = 2500)
p.add_argument('--batch_size', type=int, default = 7)
#NETWORK PARAMETERS
p.add_argument('--output_type', type = str, default = 'position') # position or pose
p.add_argument('--hidden_size', type=int, default = 3) # RNN output size
p.add_argument('--input_size', type=int, default = 4) #RNN input size: number of uwb
p.add_argument('--preprocessing_output_size', type=int, default = 3)
p.add_argument('--first_layer_output_size', type=int, default = 10)
p.add_argument('--second_layer_output_size', type=int, default = 3)
p.add_argument('--sequence_length', type=int, default = 5) # # of lstm rolling
p.add_argument('--output_size', type=int, default = 3) #final output size (RNN or softmax, etc)
#FOR TEST
p.add_argument('--load_model_dir', type=str, default="model/RiTA_wo_fcn/stacked_bi_epoch_3000/model_0_00006-17700")
p.add_argument('--test_data', type=str, default='inputs/test_data_diagonal_curve2D.csv')
p.add_argument('--output_results', type=str, default= 'results/RiTA/stack_lstm_epoch3000_17700.csv')
###########
p.add_argument('--mode', type=str, default = "train") #train or test
args = p.parse_args()

data_parser = DataPreprocessing.DataManager(args.train_data, args.sequence_length, args.input_size)
data_parser.fitDataForMinMaxScaler()

d0_data, d1_data, d2_data, d3_data = data_parser.set_range_data_for_4_uwb()

print(d0_data.shape) #Data size / sequence length / uwb num

robot_pose_gt, relative_anchor_position_gt = data_parser.set_gt_data()
print (robot_pose_gt.shape)
# print(X_data[2])
# print(X_data[-1])
# print(robot_pose_data[-1])
# print(relative_position_anchor_data[-1])
# data : size of data - sequence length + 1

tf.reset_default_graph()

ro_net = RONet(args)

#terms for learning rate decay
global_step = tf.Variable(0, trainable=False)
iter = int(len(d0_data)/args.batch_size)
num_total_steps = args.epoches*iter
ro_net.build_loss(args.lr, args.decay_rate, num_total_steps/args.decay_step)
saver = tf.train.Saver(max_to_keep = 5)

# Use simple momentum for the optimization.

###########for using tensorboard########
merged = tf.summary.merge_all()
########################################
with tf.Session() as sess:
    if (args.mode=='train'):

        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(args.board_dir, sess.graph)
        step = 0
        min_loss = 2
        tqdm_range = trange(args.epoches, desc = 'Loss', leave = True)
        for ii in tqdm_range:
            loss_of_epoch = 0
            for i in range(iter): #iter = int(len(X_data)/batch_size)
                step = step + 1
                idx = i* args.batch_size

                l, _,gt, prediction, summary = sess.run([ro_net.loss, ro_net.train, ro_net.position_gt, ro_net.pose_pred, merged ],
                                                        feed_dict={ro_net.d0_data: d0_data[idx : idx + args.batch_size],
                                                                   ro_net.d1_data: d1_data[idx : idx + args.batch_size],
                                                                   ro_net.d2_data: d2_data[idx : idx + args.batch_size],
                                                                   ro_net.d3_data: d3_data[idx : idx + args.batch_size],
                                                                   ro_net.position_gt: robot_pose_gt[idx : idx + args.batch_size]})
                writer.add_summary(summary, step)
                loss_of_epoch += l/args.batch_size
            loss_of_epoch /=iter
            if (loss_of_epoch < min_loss):
                min_loss = loss_of_epoch
                saver.save(sess, args.save_dir + 'model_'+'{0:.5f}'.format(loss_of_epoch).replace('.','_'), global_step=step)
            tqdm_range.set_description('Loss ' +'{0:.7f}'.format(loss_of_epoch)+'  ')
            tqdm_range.refresh()

#     elif (args.mode =='test'):
#    #For save diagonal data
#         saver.restore(sess, args.load_model_dir)
#    # tf.train.latest_checkpoint(
#
#         test_data = args.test_data
#         # diagonal_data = 'inputs/data_diagonal_w_big_error.csv'
#         data_parser.dir = test_data
#         X_test, Y_test = data_parser.set_data()
#         prediction = sess.run([ro_net.Y_pred], feed_dict={ro_net.X_data: X_test}) #prediction : type: list, [ [[[hidden_size]*sequence_length] ... ] ]
#
#         data_parser.write_file_data(args.output_results, prediction)

