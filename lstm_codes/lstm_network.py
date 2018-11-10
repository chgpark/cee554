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
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.sequence_length = args.sequence_length
        self.output_type = args.output_type
        self.is_multimodal = args.is_multimodal
        self.network_type = args.network_type

        if args.is_multimodal:
            self.set_placeholders_for_multimodal()
            if self.network_type == 'uni':
                self.build_RO_Net_multimodal()
            elif self.network_type == 'bi':
                self.build_RO_Net_bi_multimodal()
            elif self.network_type =='test':
                self.build_RO_Net_test()
        else:
            self.set_placeholders()
            if self.network_type == 'uni':
                self.build_RO_Net_uni()
            elif self.network_type == 'bi':
                self.build_RO_Net_bi()

    def set_placeholders(self):
        self.X_data = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.sequence_length, self.input_size])

        if self.output_type == 'position':
            self.position_gt = tf.placeholder(dtype=tf.float32,
                                            # shape=[None, 5, 3],
                                            shape=[None, 3],
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
#Preprocessing: Unidirectional, non-multimodal
##################################################

    def set_preprocessing_LSTM_for_4_uwb(self):
        with tf.variable_scope("preprocessing"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output, _state = tf.nn.dynamic_rnn(cell, self.X_data, dtype=tf.float32)

##################################################
#Preprocessing: Bidirectional, non-multimodal
##################################################

    def set_preprocessing_bi_LSTM_for_4_uwb(self):
        with tf.variable_scope("preprocessing0"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
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
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output0, _state = tf.nn.dynamic_rnn(cell, self.d0_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing1"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output1, _state = tf.nn.dynamic_rnn(cell, self.d0_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing2"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)

            self.output2, _state = tf.nn.dynamic_rnn(cell, self.d0_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing3"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
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
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            # outputs : tuple
            output0, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d0_data, dtype=tf.float32)
            self.output0_fw = output0[0]
            self.output0_bw = output0[1]

        with tf.variable_scope("preprocessing1"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output1, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d1_data, dtype=tf.float32)
            self.output1_fw = output1[0]
            self.output1_bw = output1[1]

        with tf.variable_scope("preprocessing2"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output2, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d2_data, dtype=tf.float32)
            self.output2_fw = output2[0]
            self.output2_bw = output2[1]

        with tf.variable_scope("preprocessing3"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
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
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            # outputs : tuple
            output0, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d0_data, dtype=tf.float32)
            self.output0_fw = output0[0]
            self.output0_bw = output0[1]

        with tf.variable_scope("preprocessing1"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output1, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d1_data, dtype=tf.float32)
            self.output1_fw = output1[0]
            self.output1_bw = output1[1]

        with tf.variable_scope("preprocessing2"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output2, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d2_data, dtype=tf.float32)
            self.output2_fw = output2[0]
            self.output2_bw = output2[1]

        with tf.variable_scope("preprocessing3"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output3, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d3_data, dtype=tf.float32)

            self.output3_fw = output3[0]
            self.output3_bw = output3[1]

        with tf.variable_scope("preprocessing4"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            # outputs : tuple
            output4, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d4_data, dtype=tf.float32)
            self.output4_fw = output4[0]
            self.output4_bw = output4[1]

        with tf.variable_scope("preprocessing5"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output5, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d5_data, dtype=tf.float32)
            self.output5_fw = output5[0]
            self.output5_bw = output5[1]

        with tf.variable_scope("preprocessing6"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output6, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d6_data, dtype=tf.float32)
            self.output6_fw = output6[0]
            self.output6_bw = output6[1]

        with tf.variable_scope("preprocessing7"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            # cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

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
            self.output = attention*self.output + self.output

    '''Stacked Bi-LSTM parts'''
    def set_first_layer_bi_LSTM(self):
        with tf.variable_scope("Stacked_bi_lstm1"):
            # outputs : tuple
            cell_forward1 = tf.contrib.rnn.BasicLSTMCell(num_units = self.first_layer_output_size)
            # cell_forward1 = tf.nn.rnn_cell.DropoutWrapper(cell_forward1, output_keep_prob= 0.75)
            cell_backward1 = tf.contrib.rnn.BasicLSTMCell(num_units = self.first_layer_output_size)
            # cell_backward1 = tf.nn.rnn_cell.DropoutWrapper(cell_backward1, output_keep_prob= 0.75)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, self.output, dtype=tf.float32)
            self.layer_output_fw = outputs[0]
            self.layer_output_bw = outputs[1]

    def concatenate_first_layer_output(self):
        with tf.variable_scope("First_layer_concatenation"):
            self.output = tf.concat([self.layer_output_fw, self.layer_output_bw], axis = 2)
            #shape: batch, sequence_length, self.first_layer_output_size*2

    def get_attentioned_first_layer_output(self):
        with tf.variable_scope("First_layer_attention"):
            attention = tf.nn.sigmoid(self.output)
            self.output = attention*self.output + self.output

    def set_second_layer_bi_LSTM(self):
        with tf.variable_scope("Stacked_bi_lstm2"):
            cell_forward2 = tf.contrib.rnn.BasicLSTMCell(num_units = self.second_layer_output_size)
            # cell_forward2 = tf.nn.rnn_cell.DropoutWrapper(cell_forward2, output_keep_prob= 0.8)
            cell_backward2 = tf.contrib.rnn.BasicLSTMCell(num_units = self.second_layer_output_size)
            # cell_backward2 = tf.nn.rnn_cell.DropoutWrapper(cell_backward2, output_keep_prob= 0.8)
            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward2, cell_backward2, self.output, dtype=tf.float32)
            self.layer_output_fw = outputs[0]
            self.layer_output_bw = outputs[1]

    def concatenate_second_layer_output(self):
        with tf.variable_scope("Second_layer_concatenation"):
            self.output = tf.concat([self.layer_output_fw, self.layer_output_bw], axis = 2)
            #shape: batch, sequence_length, self.first_layer_output_size*2

    def get_attentioned_second_layer_output(self):
        with tf.variable_scope("Second_layer_attention"):
            attention = tf.nn.sigmoid(self.output)
            self.output = attention*self.output + self.output

    def set_preprocessed_uni_LSTM(self):
        self.set_preprocessing_LSTM_for_4_uwb()
        self.get_attentioned_preprocessed_data()

    def set_preprocessed_bi_LSTM(self):
        self.set_preprocessing_bi_LSTM_for_4_uwb()
        self.concatenate_preprocessed_data_for_bi_LSTM()
        self.get_attentioned_preprocessed_data()

    def set_preprocessed_multimodal_LSTMs(self):
        self.set_multimodal_Preprocessing_LSTM_for_4_uwb()
        self.concatenate_preprocessed_data_for_multimodal_uni_LSTM()
        self.get_attentioned_preprocessed_data()

    def set_preprocessed_multimodal_bi_LSTMs(self):
        self.set_multimodal_Preprocessing_bi_LSTM_for_4_uwb()
        self.concatenate_preprocessed_data_for_multimodal_bi_LSTM()
        self.get_attentioned_preprocessed_data()

    def set_preprocessed_8multimodal_bi_LSTMs(self):
        self.set_multimodal_Preprocessing_bi_LSTM_for_8_uwb()
        self.concatenate_preprocessed_data_for_8multimodal_bi_LSTM()
        self.get_attentioned_preprocessed_data()

    def set_stacked_bi_LSTM_with_attention(self):
        self.set_first_layer_bi_LSTM()
        self.concatenate_first_layer_output()
        self.get_attentioned_first_layer_output()
        self.set_second_layer_bi_LSTM()
        self.concatenate_second_layer_output()
        self.get_attentioned_second_layer_output()

##################################################
            #Builing RO Nets
##################################################
    def build_RO_Net_uni(self):
        self.set_preprocessed_uni_LSTM()
        self.set_stacked_bi_LSTM_with_attention()
        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)

    def build_RO_Net_bi(self):
        self.set_preprocessed_bi_LSTM()
        self.set_stacked_bi_LSTM_with_attention()
        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)

    def build_RO_Net_multimodal(self):
        self.set_preprocessed_multimodal_LSTMs()
        self.set_stacked_bi_LSTM_with_attention()
        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)

    def build_RO_Net_bi_multimodal(self):
        self.set_preprocessed_multimodal_bi_LSTMs()
        self.set_stacked_bi_LSTM_with_attention()

        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)

    def build_RO_Net_bi_8multimodal(self):
        self.set_multimodal_Preprocessing_bi_LSTM_for_8_uwb()
        self.concatenate_preprocessed_data_for_8multimodal_bi_LSTM()
        self.get_attentioned_preprocessed_data()

        self.set_first_layer_bi_LSTM()
        self.concatenate_first_layer_output()
        self.get_attentioned_first_layer_output()

        self.set_second_layer_bi_LSTM()
        self.concatenate_second_layer_output()
        self.get_attentioned_second_layer_output()

        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)

    def build_RO_Net_test(self):
        self.set_multimodal_Preprocessing_bi_LSTM_for_8_uwb()
        self.concatenate_preprocessed_data_for_8multimodal_bi_LSTM()
        self.get_attentioned_preprocessed_data()

        self.set_first_layer_bi_LSTM()
        self.concatenate_first_layer_output()
        self.get_attentioned_first_layer_output()

        # self.set_second_layer_bi_LSTM()
        # self.concatenate_second_layer_output()
        # self.get_attentioned_second_layer_output()

        self.output = tf.reshape(self.output, [-1, self.sequence_length*self.first_layer_output_size*2])
        # self.output = tf.reshape(self.output, [-1, self.sequence_length*self.second_layer_output_size*2])
        # self.pose_pred = tf.contrib.layers.fully_connected(self.output, self.output_size)
        '''For test for all sequeneces!!'''
        fc_layer = tf.contrib.layers.fully_connected(self.output, self.sequence_length*self.output_size)
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

    def get_vector(self, sequence_input):
        a = []
        batch_size_of_seq = sequence_input.shape[0]
        sequence_size_of_seq = sequence_input.shape[1]
        for i in range(batch_size_of_seq):
            sequence = sequence_input[i]
            vector = []
            for j in range(sequence_size_of_seq - 1):
                v = sequence[j+1] - sequence[j]
                vector.append(v.tolist())
            a.append(vector)

        return np.array(a)

    def build_loss(self, lr, lr_decay_rate, lr_decay_step):
        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        batch_size = self.batch_size

        with tf.variable_scope('lstm_loss'):
            # loss = tf.losses.mean_squared_error(Y, outputs, weights=weights)#reduction=tf.losses.Reduction.MEAN)
            self.loss = tf.reduce_mean(tf.square(self.position_gt - self.pose_pred))
            tf.summary.scalar('lstm_loss', self.loss)

        # with tf.variable_scope('lstm_validation_loss'):
        #     # loss = tf.losses.mean_squared_error(Y, outputs, weights=weights)#reduction=tf.losses.Reduction.MEAN)
        #     # loss = tf.reduce_mean(tf.square(Y-outputs))
        #     self.val_loss = tf.reduce_mean(tf.square(self.position_gt - self.pose_pred))
        #     tf.summary.scalar('lstm_val_loss', self.val_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.contrib.framework.get_or_create_global_step()

            self.cur_lr = tf.train.exponential_decay(self.init_lr,
                                                     global_step=self.global_step,
                                                     decay_rate=self.lr_decay_rate,
                                                     decay_steps=self.lr_decay_step)

            tf.summary.scalar('global learning rate', self.cur_lr)

            self.train = tf.train.AdamOptimizer(learning_rate= self.cur_lr).minimize(self.loss)
            # optimizer = tf.train.AdamOptimizer(learning_rate= self.cur_lr).minimize(self.loss)
            # #Below line is for clipping. When train lstm, clipping let lstm train well
            # gvs = optimizer.compute_gradients(self.loss)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # self.train = optimizer.apply_gradients(capped_gvs)

