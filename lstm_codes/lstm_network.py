import numpy as np
import tensorflow as tf
import pprint

class RONet:
    def __init__(self, args): # batch_size, input_size,sequence_length, hidden_size, output_size):
        self.batch_size = args.batch_size
        self.input_size = args.num_uwb
        self.preprocessing_size = args.preprocessing_output_size
        self.first_layer_output_size = args.first_layer_output_size
        self.second_layer_output_size = args.second_layer_output_size
        self.output_size = args.output_size
        self.sequence_length = args.sequence_length
        self.output_type = args.output_type
        self.is_multimodal = args.is_multimodal
        self.network_type = args.network_type

        self.fc_layer_output_size = args.fc_layer_size
        self.dropout_prob = args.dropout_prob
        self.clip = args.clip

        if args.is_multimodal:
            self.set_placeholders_for_multimodal()
            if self.network_type == 'bi':
                self.build_RO_Net_bi_multimodal()
        else:
            if self.network_type == 'fc':
                self.set_placeholders_for_fc_layer()
                self.build_FC_layer()

            elif self.network_type == 'stacked_bi':
                self.set_placeholders_for_non_multimodal()
                self.set_stacked_bi_LSTM()

            elif self.network_type == 'RO':
                self.set_placeholders_for_non_multimodal()
                self.set_RO_Net()

            elif self.network_type == 'RO_wo_A':
                self.set_placeholders_for_non_multimodal()
                self.set_RO_Net_wo_A()

        self.set_loss_terms()

    def set_placeholders_for_fc_layer(self):
            self.X_data = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.input_size])

            if self.output_type == 'position':
                self.position_gt = tf.placeholder(dtype=tf.float32,
                                                shape=[None, 2],
                                                # shape=[None, 3],
                                                name='output_placeholder')

    def set_placeholders_for_non_multimodal(self):
        self.X_data = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.sequence_length, self.input_size],
                                     name='input_placeholder')

        if self.output_type == 'position':
            self.position_gt= tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.sequence_length, self.output_size],
                                         name='output_placeholder')

    def set_placeholders_for_multimodal(self):
        self.d0_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.sequence_length, 1],
                                           name='input_placeholder0')
        self.d1_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.sequence_length, 1],
                                           name='input_placeholder1')
        self.d2_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.sequence_length, 1],
                                           name='input_placeholder2')
        self.d3_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.sequence_length, 1],
                                           name='input_placeholder3')

        if self.input_size == 8:
            self.d4_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.sequence_length, 1],
                                           name='input_placeholder4')
            self.d5_data = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.sequence_length, 1],
                                               name='input_placeholder5')
            self.d6_data = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.sequence_length, 1],
                                               name='input_placeholder6')
            self.d7_data = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.sequence_length, 1],
                                               name='input_placeholder7')

        if self.output_type == 'position':
            self.position_gt = tf.placeholder(dtype=tf.float32,
                                            shape=[None,5, 3],
                                              # shape=[None, 3],
                                            name='output_placeholder')

##################################################
#Preprocessing: Bidirectional, non-multimodal
##################################################

    def set_preprocessing_bi_LSTM_for_8_uwb(self):
        with tf.variable_scope("preprocessing0"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            # outputs : tuple
            output, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.X_data, dtype=tf.float32)
            self.output_fw = output[0]
            self.output_bw = output[1]

    def concatenate_preprocessed_data_for_bi_LSTM(self):
        self.output = tf.concat([self.output_fw, self.output_bw], axis = 2)

##################################################
#Preprocessing: Unidirectional, multimodal
##################################################

    def set_multimodal_Preprocessing_LSTM_for_4_uwb(self):
        with tf.variable_scope("preprocessing0"):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output0, _state = tf.nn.dynamic_rnn(cell, self.d0_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing1"):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output1, _state = tf.nn.dynamic_rnn(cell, self.d0_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing2"):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output2, _state = tf.nn.dynamic_rnn(cell, self.d0_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing3"):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output3, _state = tf.nn.dynamic_rnn(cell, self.d0_data, dtype=tf.float32)

    def concatenate_preprocessed_data_for_multimodal_uni_LSTM(self):
        self.output = tf.concat([self.output0,
                                 self.output1,
                                 self.output2,
                                 self.output3], axis = 2)

##################################################
#Preprocessing: Bidirectional, multimodal
##################################################
    def set_multimodal_Preprocessing_bi_LSTM_for_4_uwb(self):
        with tf.variable_scope("preprocessing0"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            # outputs : tuple
            output0, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d0_data, dtype=tf.float32)
            self.output0_fw = output0[0]
            self.output0_bw = output0[1]

        with tf.variable_scope("preprocessing1"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output1, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d1_data, dtype=tf.float32)
            self.output1_fw = output1[0]
            self.output1_bw = output1[1]

        with tf.variable_scope("preprocessing2"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output2, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d2_data, dtype=tf.float32)
            self.output2_fw = output2[0]
            self.output2_bw = output2[1]

        with tf.variable_scope("preprocessing3"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output3, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d3_data, dtype=tf.float32)

            self.output3_fw = output3[0]
            self.output3_bw = output3[1]


    def concatenate_preprocessed_data_for_multimodal_bi_LSTM(self):
        self.output = tf.concat([self.output0_fw, self.output0_bw,
                                               self.output1_fw, self.output1_bw,
                                               self.output2_fw, self.output2_bw,
                                               self.output3_fw, self.output3_bw], axis = 2)

    def set_multimodal_Preprocessing_bi_LSTM_for_8_uwb(self):
        with tf.variable_scope("preprocessing0"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob=self.dropout_prob)

            # outputs : tuple
            output0, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d0_data, dtype=tf.float32)
            self.output0_fw = output0[0]
            self.output0_bw = output0[1]

        with tf.variable_scope("preprocessing1"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob = self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob = self.dropout_prob)

            output1, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d1_data, dtype=tf.float32)
            self.output1_fw = output1[0]
            self.output1_bw = output1[1]

        with tf.variable_scope("preprocessing2"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= self.dropout_prob)

            output2, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d2_data, dtype=tf.float32)
            self.output2_fw = output2[0]
            self.output2_bw = output2[1]

        with tf.variable_scope("preprocessing3"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= self.dropout_prob)

            output3, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d3_data, dtype=tf.float32)

            self.output3_fw = output3[0]
            self.output3_bw = output3[1]

        with tf.variable_scope("preprocessing4"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= self.dropout_prob)

            # outputs : tuple
            output4, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d4_data, dtype=tf.float32)
            self.output4_fw = output4[0]
            self.output4_bw = output4[1]

        with tf.variable_scope("preprocessing5"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob=self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= self.dropout_prob)

            output5, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d5_data, dtype=tf.float32)
            self.output5_fw = output5[0]
            self.output5_bw = output5[1]

        with tf.variable_scope("preprocessing6"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob=self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= self.dropout_prob)

            output6, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d6_data, dtype=tf.float32)
            self.output6_fw = output6[0]
            self.output6_bw = output6[1]

        with tf.variable_scope("preprocessing7"):
            cell_forward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= self.dropout_prob)
            cell_backward = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.preprocessing_size)
            cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= self.dropout_prob)

            output7, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d7_data, dtype=tf.float32)

            self.output7_fw = output7[0]
            self.output7_bw = output7[1]

    def concatenate_preprocessed_data_for_8multimodal_bi_LSTM(self):
        self.output = tf.concat([self.output0_fw, self.output0_bw,
                                 self.output1_fw, self.output1_bw,
                                 self.output2_fw, self.output2_bw,
                                 self.output3_fw, self.output3_bw,
                                 self.output4_fw, self.output4_bw,
                                 self.output5_fw, self.output5_bw,
                                 self.output6_fw, self.output6_bw,
                                 self.output7_fw, self.output7_bw], axis = 2)


    def get_attentioned_preprocessed_data(self):
        with tf.variable_scope("preprocessed_data_attention"):
            attention = tf.nn.sigmoid(self.output)
            self.preprocessed_output = attention*self.output + self.output

    '''Stacked Bi-LSTM parts'''
    def set_first_layer_bi_LSTM(self):
        with tf.variable_scope("Stacked_bi_lstm1"):
            # outputs : tuple
            cell_forward1 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.first_layer_output_size)
            cell_forward1 = tf.nn.rnn_cell.DropoutWrapper(cell_forward1, output_keep_prob= self.dropout_prob)
            cell_backward1 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.first_layer_output_size)
            cell_backward1 = tf.nn.rnn_cell.DropoutWrapper(cell_backward1, output_keep_prob= self.dropout_prob)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, self.preprocessed_output, dtype=tf.float32)
            self.layer_output_fw = outputs[0]
            self.layer_output_bw = outputs[1]

    def concatenate_first_layer_output(self):
        with tf.variable_scope("First_layer_concatenation"):
            self.output = tf.concat([self.layer_output_fw, self.layer_output_bw], axis = 2)
            self.output = tf.nn.relu(self.output)
            #shape: batch, sequence_length, self.first_layer_output_size*2

    def get_attentioned_first_layer_output(self):
        with tf.variable_scope("First_layer_attention"):
            attention = tf.nn.sigmoid(self.output)
            self.output = attention*self.output + self.output

    def set_second_layer_bi_LSTM(self):
        with tf.variable_scope("Stacked_bi_lstm2"):
            cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.second_layer_output_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob= self.dropout_prob)
            cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.second_layer_output_size)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob= self.dropout_prob)
            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.output, dtype=tf.float32)

            self.layer_output_fw = outputs[0]
            self.layer_output_bw = outputs[1]

    def concatenate_second_layer_output(self):
        with tf.variable_scope("Second_layer_concatenation"):
            self.output = tf.concat([self.layer_output_fw, self.layer_output_bw], axis = 2)
            self.output = tf.nn.relu(self.output)
            #shape: batch, sequence_length, self.first_layer_output_size*2

    def get_attentioned_second_layer_output(self):
        with tf.variable_scope("Second_layer_attention"):
            attention = tf.nn.sigmoid(self.output)
            self.output = attention*self.output + self.output

    def set_fc_layer_for_dynamic_len(self):
        feature_length = self.output.shape[1].value
        interval = int(feature_length / self.sequence_length)

        prediction_list = []

        for i in range(self.sequence_length):
            with tf.variable_scope("fc_{}".format(i+1)):
                fc_layer = tf.contrib.layers.fully_connected(self.output[:, i * interval:(i+1) * interval], self.fc_layer_output_size)
                fc_layer = tf.contrib.layers.fully_connected(fc_layer, 2)

                self.partial_position_pred = tf.reshape(fc_layer, [-1, 1, 2])

            prediction_list.append(self.partial_position_pred)


        self.pose_pred = tf.concat(prediction_list, axis=1)

    def set_fc_layer_for_seq_len_5(self):
        shape_1 = self.output.shape[1].value
        interval = int(shape_1 / self.sequence_length)


        with tf.variable_scope("fc1"):
            fc_layer = tf.contrib.layers.fully_connected(self.output[:, :interval], self.fc_layer_output_size)
            fc_layer = tf.contrib.layers.fully_connected(fc_layer, 2)

            self.position_pred1 = tf.reshape(fc_layer, [-1, 1, 2])

        with tf.variable_scope("fc2"):
            fc_layer = tf.contrib.layers.fully_connected(self.output[:, interval:2*interval], self.fc_layer_output_size)
            fc_layer = tf.contrib.layers.fully_connected(fc_layer, 2)

            self.position_pred2 = tf.reshape(fc_layer, [-1, 1, 2])

        with tf.variable_scope("fc3"):
            fc_layer = tf.contrib.layers.fully_connected(self.output[:, 2*interval:3*interval], self.fc_layer_output_size)
            fc_layer = tf.contrib.layers.fully_connected(fc_layer, 2)

            self.position_pred3 = tf.reshape(fc_layer, [-1, 1, 2])

        with tf.variable_scope("fc4"):
            fc_layer = tf.contrib.layers.fully_connected(self.output[:, 3*interval:4*interval], self.fc_layer_output_size)
            fc_layer = tf.contrib.layers.fully_connected(fc_layer, 2)

            self.position_pred4 = tf.reshape(fc_layer, [-1, 1, 2])

        with tf.variable_scope("fc5"):
            fc_layer = tf.contrib.layers.fully_connected(self.output[:, 4*interval:5*interval], self.fc_layer_output_size)
            fc_layer = tf.contrib.layers.fully_connected(fc_layer, 2)

            self.position_pred5 = tf.reshape(fc_layer, [-1, 1, 2])

        self.pose_pred = tf.concat([self.position_pred1, self.position_pred2, self.position_pred3, self.position_pred4, self.position_pred5], axis=1)

##################################################
            #Builing RiTA's paper
##################################################

    def set_stacked_bi_LSTM(self):
        with tf.variable_scope("Stacked_bi_lstm1"):
            # outputs : tuple
            first_layer_output_num = 100
            cell_forward1 = tf.contrib.rnn.BasicLSTMCell(num_units=first_layer_output_num)
            cell_backward1 = tf.contrib.rnn.BasicLSTMCell(num_units=first_layer_output_num)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, self.X_data,
                                                               dtype=tf.float32)
            # outputs = tf.concat([outputs[0], outputs[1]], axis=1)
            outputs = tf.concat([outputs[0], outputs[1]], axis=2)

        with tf.variable_scope("Stacked_bi_lstm2"):
            cell_forward2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_size)
            cell_backward2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_size)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward2, cell_backward2, outputs, dtype=tf.float32)

            outputs = tf.concat([outputs[0], outputs[1]], axis=2)

        self.output = tf.reshape(outputs, [-1, self.sequence_length*2*self.output_size])

        self.set_fc_layer_for_dynamic_len()


##################################################
            #Builing RO Nets
##################################################
    def set_RO_Net(self):
        with tf.variable_scope("Stacked_bi_lstm1"):
            self.set_preprocessing_bi_LSTM_for_8_uwb()
            self.concatenate_preprocessed_data_for_bi_LSTM()
            self.get_attentioned_preprocessed_data()
            self.set_first_layer_bi_LSTM()
            self.concatenate_first_layer_output()

            self.get_attentioned_first_layer_output()

            self.set_second_layer_bi_LSTM()
            self.concatenate_second_layer_output()

            self.get_attentioned_second_layer_output()

            # self.output = tf.reshape(self.output, [-1, self.sequence_length*self.first_layer_output_size*2])
            self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
            print(self.output.shape[1])
            print("hello!")
            '''For test for all sequeneces!!'''
            # self.set_fc_layer_for_seq_len_5()
            self.set_fc_layer_for_dynamic_len()

    def set_RO_Net_wo_A(self):
        with tf.variable_scope("Stacked_bi_lstm1"):
            self.set_preprocessing_bi_LSTM_for_8_uwb()
            self.concatenate_preprocessed_data_for_bi_LSTM()
            # self.get_attentioned_preprocessed_data()
            self.preprocessed_output = self.output
            self.set_first_layer_bi_LSTM()
            self.concatenate_first_layer_output()

            # self.get_attentioned_preprocessed_data()

            self.set_second_layer_bi_LSTM()
            self.concatenate_second_layer_output()

            # self.get_attentioned_preprocessed_data()

            # self.output = tf.reshape(self.output, [-1, self.sequence_length*self.first_layer_output_size*2])
            self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
            print(self.output.shape[1])
            print("hello!")
            '''For test for all sequeneces!!'''
            # self.set_fc_layer_for_seq_len_5()
            self.set_fc_layer_for_dynamic_len()

    def build_RO_Net_bi_multimodal(self):
        self.set_multimodal_Preprocessing_bi_LSTM_for_8_uwb()
        self.concatenate_preprocessed_data_for_8multimodal_bi_LSTM()
        self.get_attentioned_preprocessed_data()
        self.set_first_layer_bi_LSTM()
        self.concatenate_first_layer_output()
        self.get_attentioned_first_layer_output()
        self.set_second_layer_bi_LSTM()
        self.concatenate_second_layer_output()

        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        # self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        # self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)
        '''For test for all sequeneces!!'''
        fc_layer = tf.contrib.layers.fully_connected(self.output, 512*5)
        fc_layer = tf.contrib.layers.fully_connected(self.output, 256*5)
        fc_layer = tf.contrib.layers.fully_connected(fc_layer, self.sequence_length*self.output_size)
        self.pose_pred = tf.reshape(fc_layer, [-1, self.sequence_length, self.output_size])

    def build_FC_layer(self):
        num_layer = 100
        with tf.variable_scope("FC_layer"):
            fc_layer = tf.contrib.layers.fully_connected(self.X_data, num_layer)
            fc_layer = tf.contrib.layers.fully_connected(fc_layer, num_layer)

            self.pose_pred = tf.contrib.layers.fully_connected(fc_layer, self.output_size)

    def build_RO_Net_bi_multimodal_one_layer(self):

        self.set_multimodal_Preprocessing_bi_LSTM_for_8_uwb()
        self.concatenate_preprocessed_data_for_8multimodal_bi_LSTM()
        self.get_attentioned_preprocessed_data()

        with tf.variable_scope("Stacked_bi_lstm_1st_layer"):
            # outputs : tuple
            cell_forward1 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.first_layer_output_size)
            cell_forward1 = tf.nn.rnn_cell.DropoutWrapper(cell_forward1, output_keep_prob= self.dropout_prob)
            cell_backward1 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units = self.first_layer_output_size)
            cell_backward1 = tf.nn.rnn_cell.DropoutWrapper(cell_backward1, output_keep_prob= self.dropout_prob)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, self.preprocessed_output, dtype=tf.float32)
            self.layer_output_fw = outputs[0]
            self.layer_output_bw = outputs[1]

        self.concatenate_first_layer_output()

        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.first_layer_output_size*2])
        # self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        # self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)
        '''For test for all sequeneces!!'''
        fc_layer = tf.contrib.layers.fully_connected(self.output, self.second_layer_output_size)
        fc_layer = tf.contrib.layers.fully_connected(fc_layer, self.sequence_length*self.output_size)
        self.pose_pred = tf.reshape(fc_layer, [-1, self.sequence_length, self.output_size])


##################################################
            #Building loss
##################################################
    def build_smooth_L1_loss(self):
        criteria = tf.reduce_mean((self.position_gt - self.pose_pred))
        if criteria > 0.2:
            return criteria
        else:
            return tf.reduce_mean(tf.square(self.position_gt - self.pose_pred))


    def set_loss_terms(self):
        print ("Bulding loss terms...")
        # self.vec_pose_pred = tf.subtract(self.pose_pred[:, 1:, :], self.pose_pred[:, :-1, :])
        # self.vec_position_gt = tf.subtract(self.position_gt[:, 1:, :], self.position_gt[:, :-1, :])
        # epsilon = tf.constant(0.0000000000001, dtype= tf.float32)
        # norm_gt = tf.add(tf.norm(self.position_gt, axis = 2) , epsilon)
        # norm_pred = tf.add(tf.norm(self.pose_pred, axis = 2) , epsilon)
        # self.magnitude_of_pose_pred = tf.reduce_mean(tf.square(self.pose_pred))
        #
        # self.direction_error_btw_gt_and_pred = tf.reduce_mean(1 - tf.divide((tf.reduce_sum(self.position_gt* self.pose_pred, axis = 2)), norm_gt*norm_pred))

        self.error_btw_gt_and_pred = tf.reduce_mean(tf.square(self.position_gt - self.pose_pred))

        print ("Complete!")
    def build_loss(self, lr, lr_decay_rate, lr_decay_step):


        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        batch_size = self.batch_size

        with tf.variable_scope('lstm_loss'):
            self.loss = self.error_btw_gt_and_pred
            tf.summary.scalar('lstm_loss', self.loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.cur_lr = tf.train.exponential_decay(self.init_lr,
                                                     global_step=self.global_step,
                                                     decay_rate=self.lr_decay_rate,
                                                     decay_steps=self.lr_decay_step)

            tf.summary.scalar('global learning rate', self.cur_lr)

            self.optimizer = tf.train.AdamOptimizer(learning_rate= self.cur_lr)
            '''Gradient clipping parts'''
            # gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            # self.optimize = self.optimizer.apply_gradients(zip(gradients, variables))
            self.optimize = self.optimizer.minimize(self.loss, global_step=self.global_step)

