import tensorflow as tf
import pprint

class RONet:
    def __init__(self, args): # batch_size, input_size,sequence_length, hidden_size, output_size):
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.preprocessing_size = args.preprocessing_output_size
        self.first_layer_output_size = args.first_layer_output_size
        self.second_layer_output_size = args.second_layer_output_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.sequence_length = args.sequence_length
        self.output_type = args.output_type
        self.set_placeholders()

        self.buildRONet()

    def set_placeholders(self):
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
                                            shape=[None, 3],
                                            name='output_placeholder')

    def set_multimodal_PreprocessingLSTM_for_4_uwb(self):
        with tf.variable_scope("preprocessing0"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            # outputs : tuple
            output0, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d0_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing1"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output1, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d1_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing2"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output2, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d2_data, dtype=tf.float32)

        with tf.variable_scope("preprocessing3"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.preprocessing_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            output3, _state = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.d3_data, dtype=tf.float32)

            return output0, output1, output2, output3

    def getPreprocessedData(self):
        output0, output1, output2, output3 = self.set_multimodal_PreprocessingLSTM_for_4_uwb()
        outputs_array = []
        for i in range(self.batch_size): # shape of output0[0]: (num_data, seq_length, hidden_size)
            concatnated_output = tf.concat([output0[0][i], output0[1][i],
                                            output1[0][i], output1[1][i],
                                            output2[0][i], output2[1][i],
                                            output3[0][i], output3[1][i]], axis = 1)
            outputs_array.append(concatnated_output)

        outputs_array = self.getAttentionedOutput(outputs_array)
        return outputs_array

    def getAttentionedOutput(self, tensor):
        attention = tf.nn.sigmoid(tensor)
        attentioned_tensor = attention*tensor

        return attentioned_tensor + tensor

    def getConcatenatedTensor(self, outputs):
        concatenated_tensor = []
        for i in range(self.batch_size):
            concatenated_output = tf.concat([outputs[0][i], outputs[1][i]], axis = 1)
            concatenated_tensor.append(concatenated_output)

        # concatenated_tensor = tf.convert_to_tensor(concatenated_tensor)
        return concatenated_tensor

    def setStackedBiLSTMwithAttention(self, input_data):
        with tf.variable_scope("Stacked_bi_lstm1"):
            # outputs : tuple
            first_layer_output_num = 10
            cell_forward1 = tf.contrib.rnn.BasicLSTMCell(num_units = self.first_layer_output_size)
            cell_backward1 = tf.contrib.rnn.BasicLSTMCell(num_units = self.first_layer_output_size)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, input_data, dtype=tf.float32)
            outputs = self.getConcatenatedTensor(outputs)
            #shape: batch, sequence_length, self.first_layer_output_size*2
            attentioned_outputs = self.getAttentionedOutput(outputs)

        with tf.variable_scope("Stacked_bi_lstm2"):
            cell_forward2 = tf.contrib.rnn.BasicLSTMCell(num_units = self.second_layer_output_size)
            cell_backward2 = tf.contrib.rnn.BasicLSTMCell(num_units = self.second_layer_output_size)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward2, cell_backward2, attentioned_outputs, dtype=tf.float32)
            outputs = self.getConcatenatedTensor(outputs)
            # outputs = tf.convert_to_tensor(outputs)

            #shape: batch, sequence_length, self.second_layer_output_size*2
            attentioned_outputs = self.getAttentionedOutput(outputs)
            return attentioned_outputs

    def buildRONet(self):
        input_to_LSTM = self.getPreprocessedData() #shape: batch, sequence_length, intput_size*2*self.preprocessing_output_size
        lstm_output = self.setStackedBiLSTMwithAttention(input_to_LSTM)
        position = lstm_output[:, -1, :]
        self.pose_pred = tf.contrib.layers.fully_connected(position, 3)

    def build_loss(self, lr, lr_decay_rate, lr_decay_step):
        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        batch_size = self.batch_size

        with tf.variable_scope('lstm_loss'):
            # loss = tf.losses.mean_squared_error(Y, outputs, weights=weights)#reduction=tf.losses.Reduction.MEAN)
            # loss = tf.reduce_mean(tf.square(Y-outputs))
            self.loss = tf.reduce_sum(tf.square(self.position_gt - self.pose_pred))
            tf.summary.scalar('lstm_loss', self.loss)

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
