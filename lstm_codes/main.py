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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
tf.set_random_seed(777)  # reproducibilityb
# hyper parameters
p =argparse.ArgumentParser()
#FOR TRAIN
p.add_argument('--train_data', type=str, default="train_3D_zigzag_1.csv")
p.add_argument('--val_data', type=str, default="./inputs/spiral_3D.csv")
p.add_argument('--board_dir', type=str, default="/home/shapelim/RONet_result/board/multimodal/stacked_bi_e2500_lr0_02_1/")
p.add_argument('--save_dir', type=str, default="/home/shapelim/RONet_result/model/multimodal/stacked_bi_e2500_lr0_02_1/")

p.add_argument('--lr', type=float, default = 0.02)
p.add_argument('--decay_rate', type=float, default = 0.7)
p.add_argument('--decay_step', type=int, default = 7)
p.add_argument('--epoches', type=int, default = 10)
p.add_argument('--batch_size', type=int, default = 200)

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
p.add_argument('--load_model_dir', type=str, default="/home/shapelim/RONet_result/model/multimodal/stacked_bi_epoch_2500_lr_0_02/model_0_00013-132500")
p.add_argument('--test_data', type=str, default='inputs/poly_3D_zigzag.csv')
p.add_argument('--output_results', type=str, default= 'results/multimodal/multimodal_poly.csv')
###########
p.add_argument('--mode', type=str, default = "train") #train or test
args = p.parse_args()

print ("Loading train data...")
data_parser = DataPreprocessing.DataManager(args.train_data, args.sequence_length, args.input_size)
data_parser.fitDataForMinMaxScaler()
print ("Complete!")
d0_data, d1_data, d2_data, d3_data = data_parser.set_range_data_for_4_uwb()
robot_pose_gt, relative_anchor_position_gt = data_parser.set_gt_data()


print(d0_data.shape) #Data size / sequence length / uwb num
'''For validation'''

print ("Loading val data...")
data_parser.set_dir(args.val_data)
val_d0_data, val_d1_data, val_d2_data, val_d3_data = data_parser.set_range_data_for_4_uwb()
val_robot_pose_gt, _ = data_parser.set_gt_data()
print ("Complete!")
# d0_data, d1_data, d2_data, d3_data, robot_pose_gt = data_parser.suffle_array_in_the_same_order(d0_data, d1_data, d2_data, d3_data, robot_pose_gt)

# print(X_data[2])
# print(X_data[-1])
# print(robot_pose_data[-1])
# print(relative_position_anchor_data[-1])
# data : size of data - sequence length + 1

writer_val = tf.summary.FileWriter('./logs/val') #, sess.graph)
writer_train = tf.summary.FileWriter('./logs/train') #, sess.graph)

tf.reset_default_graph()
ro_net = RONet(args)

#terms for learning rate decay
global_step = tf.Variable(0, trainable=False)
iter = int(len(d0_data)/args.batch_size)
num_total_steps = args.epoches*iter
ro_net.build_loss(args.lr, args.decay_rate, num_total_steps/args.decay_step)
saver = tf.train.Saver(max_to_keep = 5)

# Use simple momentum for the optimization.

# COUNT PARAMS
total_num_parameters = 0
for variable in tf.trainable_variables():
    total_num_parameters += np.array(variable.get_shape().as_list()).prod()
print("number of trainable parameters: {}".format(total_num_parameters))

###########for using tensorboard########

merged = tf.summary.merge_all()
########################################

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

with tf.Session() as sess:
    if (args.mode=='train'):

        sess.run(tf.global_variables_initializer())



        step = 0
        min_loss = 2
        tqdm_range = trange(args.epoches, desc = 'Loss', leave = True)
        for ii in tqdm_range:
            loss_of_epoch = 0
            loss_of_val = 0
            for i in range(iter): #iter = int(len(X_data)/batch_size)
                step = step + 1
                idx = i* args.batch_size

                l, _, summary = sess.run([ro_net.loss, ro_net.train, merged],
                                                        feed_dict={ro_net.d0_data: d0_data[idx : idx + args.batch_size],
                                                                   ro_net.d1_data: d1_data[idx : idx + args.batch_size],
                                                                   ro_net.d2_data: d2_data[idx : idx + args.batch_size],
                                                                   ro_net.d3_data: d3_data[idx : idx + args.batch_size],
                                                                   ro_net.position_gt: robot_pose_gt[idx : idx + args.batch_size]})
                writer_train.add_summary(summary, step)
                writer_train.flush()

                loss_of_epoch += l

                val_l, summary = sess.run([ro_net.loss, merged], feed_dict={ro_net.d0_data: val_d0_data,
                                                      ro_net.d1_data: val_d1_data,
                                                      ro_net.d2_data: val_d2_data,
                                                      ro_net.d3_data: val_d3_data,
                                                      ro_net.position_gt: val_robot_pose_gt} )
                writer_val.add_summary(summary, step)
                writer_val.flush()

                loss_of_val += val_l

            loss_of_epoch /= iter
            loss_of_val /= iter
            if (loss_of_epoch < min_loss):
                min_loss = loss_of_epoch
                saver.save(sess, args.save_dir + 'model_'+'{0:.5f}'.format(loss_of_epoch).replace('.','_'), global_step=step)
            tqdm_range.set_description('train ' +'{0:.7f}'.format(loss_of_epoch)+' /val '+'{0:.7f}'.format(loss_of_val) )
            tqdm_range.refresh()

    elif (args.mode =='test'):
   #For save diagonal data
        saver.restore(sess, args.load_model_dir)
        print ("Load success.")


        data_parser.set_dir(args.test_data)
        d0_data, d1_data, d2_data, d3_data = data_parser.set_range_data_for_4_uwb()
        prediction = sess.run(ro_net.pose_pred, feed_dict={ro_net.d0_data: d0_data,
                                                          ro_net.d1_data: d1_data,
                                                          ro_net.d2_data: d2_data,
                                                          ro_net.d3_data: d3_data}) #prediction : type: list, [ [[[hidden_size]*sequence_length] ... ] ]

        print ((prediction))
        data_parser.inverse_transform_by_train_data(prediction)
        data_parser.write_file_data(args.output_results)

