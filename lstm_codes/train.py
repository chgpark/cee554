import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from lstm_network import RONet
import numpy as np
import DataPreprocessing
from tqdm import tqdm, trange
import os
import argparse
import csv
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
tf.set_random_seed(777)  # reproducibilityb
# hyper parameters
p =argparse.ArgumentParser()

#FOR TRAIN
p.add_argument('--mode', type=str, default='train')
p.add_argument('--train_data', type=str, default="/home/shapelim/RONet/RO_train/")
p.add_argument('--val_data', type=str, default="/home/shapelim/RONet/RO_val/")
p.add_argument('--save_dir', type=str, default="/home/shapelim/RONet/test_fc_2/")

p.add_argument('--lr', type=float, default = 0.001)
p.add_argument('--decay_rate', type=float, default = 0.7)
p.add_argument('--decay_step', type=int, default = 5)
p.add_argument('--epoches', type=int, default = 1500)
p.add_argument('--batch_size', type=int, default = 6450) #11257)

#NETWORK PARAMETERS
p.add_argument('--output_type', type = str, default = 'position') # position or pose
p.add_argument('--num_uwb', type=int, default = 8) # RNN input size: number of uwb
p.add_argument('--preprocessing_output_size', type=int, default = 512)
p.add_argument('--first_layer_output_size', type=int, default = 256)
# Second: not in use
p.add_argument('--second_layer_output_size', type=int, default = 128)
p.add_argument('--fc_layer_size', type=int, default=1024)
p.add_argument('--sequence_length', type=int, default = 8) # # of lstm rolling
p.add_argument('--output_size', type=int, default = 2) # We only infer x, y
'''
network_type
is_multimodal == True => stacked_bi
is_multimodal == False => fc / stacked_bi / RO / RO_wo_A
'''
p.add_argument('--is_multimodal', type=bool, default = False) #True / False
p.add_argument('--network_type', type=str, default = 'fc')
p.add_argument('--clip', type=float, default = 5.0)
p.add_argument('--dropout_prob', type=float, default = 1.0)

p.add_argument('--gpu', type=str, default='0')



args = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu #,1,2,3'

data_parser = DataPreprocessing.DataManager(args)
data_parser.fit_all_data()

if args.is_multimodal:
    print ("Loading train data for multimodal...")
    # data_parser.set_data_for_8multimodal()
    data_parser.set_data_for_8multimodal_all_sequences()
    d0_data, d1_data, d2_data, d3_data, d4_data, d5_data, d6_data, d7_data = data_parser.get_range_data_for_8multimodal()
    robot_position_gt, robot_quaternion_gt = data_parser.get_gt_data()
    print ("Complete!")
    print(d0_data.shape) #Data size / sequence length / uwb num

    print ("Loading val data...")
    data_parser.set_dir(args.val_data)
    data_parser.set_all_target_data_list(generating_grid=True)
    data_parser.transform_all_data()
    # data_parser.set_data_for_8multimodal()
    data_parser.set_data_for_8multimodal_all_sequences()
    val_d0_data, val_d1_data, val_d2_data, val_d3_data, val_d4_data, val_d5_data, val_d6_data, val_d7_data = data_parser.get_range_data_for_8multimodal()
    val_robot_position_gt, val_robot_quaternion_gt = data_parser.get_gt_data()
    print ("Complete!")


    print (robot_position_gt.shape)

    writer_val = tf.summary.FileWriter(args.save_dir + '/board/val') #, sess.graph)
    writer_train = tf.summary.FileWriter(args.save_dir + '/board/train') #, sess.graph)

    tf.reset_default_graph()
    ro_net = RONet(args)

    #For Generating grid!!!
    # ro_net.get_scale_for_round(data_parser.scaler_for_prediction.scale_)
    # ro_net.round_predicted_position()

    #terms for learning rate decay
    global_step = tf.Variable(0, trainable=False)

    iter = int(len(d0_data)/args.batch_size)
    num_total_steps = args.epoches*iter
    ro_net.build_loss(args.lr, args.decay_rate, num_total_steps/args.decay_step)
    saver = tf.train.Saver(max_to_keep = 3)

    # Use simple momentum for the optimization.

    # COUNT PARAMS
    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(total_num_parameters))


    ###########for using tensorboard########
    merged = tf.summary.merge_all()
    ########################################
    #
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    #
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        min_loss = 1000000000
        tqdm_range = trange(args.epoches, desc = 'Loss', leave = True)
        before_idx = 0
        next_idx = 0
        for ii in tqdm_range:
            loss_of_epoch = 0
            loss_of_val = 0
            i = 1

            d0_data, d1_data, d2_data, d3_data, d4_data, d5_data, d6_data, d7_data, robot_position_gt = data_parser.suffle_array_in_the_same_order(d0_data, d1_data, d2_data, d3_data, d4_data, d5_data, d6_data, d7_data, robot_position_gt)
            for i in range(iter): #iter = int(len(X_data)/batch_size)
                step = step + 1
                idx = i* args.batch_size
                l, _, summary = sess.run([ro_net.loss, ro_net.optimize, merged],
                                        feed_dict={ro_net.d0_data: d0_data[idx: idx + args.batch_size],
                                                   ro_net.d1_data: d1_data[idx: idx + args.batch_size],
                                                   ro_net.d2_data: d2_data[idx: idx + args.batch_size],
                                                   ro_net.d3_data: d3_data[idx: idx + args.batch_size],
                                                   ro_net.d4_data: d4_data[idx: idx + args.batch_size],
                                                   ro_net.d5_data: d5_data[idx: idx + args.batch_size],
                                                   ro_net.d6_data: d6_data[idx: idx + args.batch_size],
                                                   ro_net.d7_data: d7_data[idx: idx + args.batch_size],
                                                   ro_net.position_gt: robot_position_gt[idx: idx + args.batch_size]})
                writer_train.add_summary(summary, step)
                writer_train.flush()

                loss_of_epoch += l

            loss_of_val, summary = sess.run([ro_net.loss, merged], feed_dict={ro_net.d0_data: val_d0_data,
                                                                              ro_net.d1_data: val_d1_data,
                                                                              ro_net.d2_data: val_d2_data,
                                                                              ro_net.d3_data: val_d3_data,
                                                                              ro_net.d4_data: val_d4_data,
                                                                              ro_net.d5_data: val_d5_data,
                                                                              ro_net.d6_data: val_d6_data,
                                                                              ro_net.d7_data: val_d7_data,
                                                                              ro_net.position_gt: val_robot_position_gt})
            writer_val.add_summary(summary, step)
            writer_val.flush()


            loss_of_epoch /= iter
            if (loss_of_epoch < min_loss):
                min_loss = loss_of_epoch
                saver.save(sess, args.save_dir + 'model_'+'{0:.5f}'.format(loss_of_epoch).replace('.', '_'), global_step=step)
            tqdm_range.set_description('train ' +'{0:.7f}'.format(loss_of_epoch)+' | val '+'{0:.7f}'.format(loss_of_val))
            tqdm_range.refresh()
        f = open(args.save_dir + "final_losses.txt", 'w')
        f.write("loss : " + str(loss_of_epoch) + '\n')
        f.close()


else:
    '''train for non-multimodal case'''
    print ("Loading train data for non multimodal...")
    if args.network_type == 'fc':
        data_parser.set_train_data_for_fc_layer()
    else:
        data_parser.set_train_data()

    X_data = data_parser.get_range_data_for_nonmultimodal()
    robot_position_gt = data_parser.get_gt_data()
    print ("Complete!")
    print (X_data.shape, robot_position_gt.shape) #Data size / sequence length / uwb num or (batch, uwb_num)

    print ("Loading val data...")
    if args.network_type == 'fc':
        data_parser.set_val_data_for_fc_layer()
    else:
        data_parser.set_val_data()

    val_X_data = data_parser.get_range_data_for_nonmultimodal()
    val_robot_position_gt = data_parser.get_gt_data()
    print ("Complete!")

    writer_val = tf.summary.FileWriter(args.save_dir + '/board/val') #, sess.graph)
    writer_train = tf.summary.FileWriter(args.save_dir + '/board/train') #, sess.graph)

    tf.reset_default_graph()
    ro_net = RONet(args)

    #terms for learning rate decay
    global_step = tf.Variable(0, trainable=False)

    iter = int(len(X_data)/args.batch_size)

    num_total_steps = args.epoches*iter
    ro_net.build_loss(args.lr, args.decay_rate, num_total_steps/args.decay_step)
    saver = tf.train.Saver(max_to_keep = 3)

    # COUNT PARAMS
    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(total_num_parameters))

    ###########for using tensorboard########
    merged = tf.summary.merge_all()
    ########################################
    #
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    #
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        min_loss = 1000000000
        tqdm_range = trange(args.epoches, desc = 'Loss', leave = True)
        before_idx = 0
        next_idx = 0
        for ii in tqdm_range:
            loss_of_epoch = 0
            loss_of_val = 0
            i = 1

            # d0_data, d1_data,
            X_data, robot_position_gt = data_parser.suffle_array_in_the_same_order(X_data, robot_position_gt)
            for i in range(iter): #iter = int(len(X_data)/batch_size)
                step = step + 1
                idx = i* args.batch_size
                l, _, summary = sess.run([ro_net.loss, ro_net.optimize, merged],
                                        feed_dict={ro_net.X_data: X_data[idx: idx + args.batch_size],
                                                   ro_net.position_gt: robot_position_gt[idx: idx + args.batch_size]})
                writer_train.add_summary(summary, step)
                writer_train.flush()

                loss_of_epoch += l

            loss_of_val, summary = sess.run([ro_net.loss, merged], feed_dict={ro_net.X_data: val_X_data,
                                                                              ro_net.position_gt: val_robot_position_gt})
            writer_val.add_summary(summary, step)
            writer_val.flush()

            loss_of_epoch /= iter
            if (loss_of_val < min_loss):
                min_loss = loss_of_val
                saver.save(sess, args.save_dir + 'model_'+'{0:.5f}'.format(loss_of_val).replace('.', '_'), global_step=step)
            tqdm_range.set_description('train ' +'{0:.7f}'.format(loss_of_epoch)+' | val '+'{0:.7f}'.format(loss_of_val))
            tqdm_range.refresh()
        f = open(args.save_dir + "final_losses.txt", 'w')
        f.write("loss : " + str(loss_of_epoch) + '\n')
        f.close()


