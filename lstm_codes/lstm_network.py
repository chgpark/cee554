import tensorflow as tf

class LSTM:
    def __init__(self, args): # batch_size, input_size,sequence_length, hidden_size, output_size):
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.sequence_length = args.sequence_length
        self.networks = args.network_model
        self.X_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.sequence_length, self.input_size],
                                           name='input_placeholder')
        self.Y_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, self.sequence_length,self.hidden_size],
                                        name='output_placeholder')

        self.build_model()
    def setLSTM(self):
        with tf.variable_scope("unidirectional_lstm"):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_size)
            # outputs : tuple

            return tf.nn.dynamic_rnn(cell, self.X_data, dtype=tf.float32)

    def setGRU(self):
        with tf.variable_scope("GRU"):
            cell = tf.contrib.rnn.GRUCell(num_units = self.hidden_size)
            # outputs : tuple

            return tf.nn.dynamic_rnn(cell, self.X_data, dtype=tf.float32)
    def setStackedBiLSTM(self):
        with tf.variable_scope("Stacked_bi_lstm1"):
            # outputs : tuple
            first_layer_output_num = 100
            cell_forward1 = tf.contrib.rnn.BasicLSTMCell(num_units= first_layer_output_num)
            cell_backward1 = tf.contrib.rnn.BasicLSTMCell(num_units=first_layer_output_num)

            # outputs : tuple
            outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_forward1, cell_backward1, self.X_data, dtype=tf.float32)
            # outputs = tf.concat([outputs[0], outputs[1]], axis=1)
            outputs = tf.add(outputs[0], outputs[1])

        with tf.variable_scope("Stacked_bi_lstm2"):
            cell_forward2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
            cell_backward2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)

            # outputs : tuple
            return tf.nn.bidirectional_dynamic_rnn(cell_forward2, cell_backward2, outputs, dtype=tf.float32)

    def setStackedBiLSTM_by_MultiRNNCell(self):
        with tf.variable_scope("MultiRNNCell"):
            # outputs : tuple
            layer_output_num = 100
            cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units= layer_output_num)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units= layer_output_num)

            lstm_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([cell_fw]*2)
            lstm_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([cell_bw]*2)

            # outputs : tuple
            return tf.nn.bidirectional_dynamic_rnn(lstm_fw_multicell, lstm_bw_multicell, self.X_data, dtype =tf.float32)   # def setStackedBiLSTM_test(self):
    #     with tf.variable_scope("Stacked_test"):
    #         # outputs : tuple
    #         first_layer_output_num =5
    #         cell_fw1 = tf.contrib.rnn.BasicLSTMCell(num_units= first_layer_output_num)
    #         cell_bw1 = tf.contrib.rnn.BasicLSTMCell(num_units=first_layer_output_num)
    #
    #         cell_fw2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
    #         cell_bw2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
    #         cell_fw = [cell_fw1, cell_fw2]
    #         cell_bw = [cell_bw1, cell_bw2]
    #
    #         # outputs : tuple
    #         return tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, self.X_data, dtype = tf.float32)


    # def setEncoderDecoderModel(self):
    #         encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.hidden_size)
    #         #   encoder_outputs: [max_time, batch_size, num_units]
    #         #   encoder_state: [batch_size, num_units]
    #         encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.X_data, dtype = tf.float32)
    #             # sequence_length=source_sequence_length, time_major=True)
    #
    #         # Build RNN cell
    #         decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.hidden_size)
    #         attention = tf.contrib.seq2seq.LuongAttention()
    #         # Helper
    #         helper = tf.contrib.seq2seq.TrainingHelper(
    #             decoder_emb_inp, decoder_lengths, time_major=True)
    #         # Decoder
    #         decoder = tf.contrib.seq2seq.BasicDecoder(
    #             decoder_cell, helper, encoder_state,
    #             output_layer=projection_layer)
    #         # Dynamic decoding
    #         outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
    #         logits = outputs.rnn_output
    #
    #         # attention_states: [batch_size, max_time, num_units]
    #
    #         attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
    #
    #         # Create an attention mechanism
    #         attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    #         num_units, attention_states,
    #         memory_sequence_length=source_sequence_length)
    #
    #         decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    #         decoder_cell, attention_mechanism,
    #         attention_layer_size=num_units)

    def setBiLSTM(self):
        with tf.variable_scope("bidirectional_lstm"):
            cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_size)
            #cell_forward = tf.nn.rnn_cell.DropoutWrapper(cell_forward, output_keep_prob= 0.7)
            cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_size)
            #cell_backward = tf.nn.rnn_cell.DropoutWrapper(cell_backward, output_keep_prob= 0.7)

            # outputs : tuple
            return tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.X_data, dtype=tf.float32)

    def build_model(self):
            if (self.networks == 'bi'):
                outputs, _states = self.setBiLSTM()
                #outputs = tf.concat([outputs[0], outputs[1]], axis=1)
                outputs = tf.add(outputs[0], outputs[1])
                print ("Bidirectional LSTM")

            elif (self.networks == 'stack'):

                outputs, _states = self.setStackedBiLSTM()
                outputs = tf.add(outputs[0], outputs[1])
                print ("Stacked Bidirectional LSTM")

            elif (self.networks =='uni'):
                outputs, _states = self.setLSTM()
                print ("Unidirectional LSTM")
            elif (self.networks =='gru'):
                outputs, _states = self.setGRU()
                print ("GRU")

            # outputs = tf.contrib.layers.fully_connected(X_for_fc, 100, activation_fn=None)
            self.Y_pred = outputs[:,-1, :]

    def build_model_w_FCN(self):
            if (self.networks == 'bi'):
                outputs, _states = self.setBidirectionalLSTM()
                outputs = tf.concat([outputs[0], outputs[1]], axis=1)
                ## FC layer
                X_for_fc = tf.reshape(outputs, [-1, self.hidden_size * 2])  # -1 for flatten
                print ("Bidirectional LSTM")
            elif (self.networks =='uni'):
                outputs, _states = self.setUnidirectionalLSTM()
                X_for_fc = tf.reshape(outputs, [-1, self.hidden_size])  # -1 for flatten
                print ("Unidirectional LSTM")

            # outputs = tf.contrib.layers.fully_connected(X_for_fc, 100, activation_fn=None)
            Y_pred = tf.contrib.layers.fully_connected(X_for_fc, 100)
            Y_pred = tf.contrib.layers.fully_connected(Y_pred, self.output_size, activation_fn=None)

            # Y_pred = tf.reshape(Y_pred, [batch_size, sequence_length, num_classes])
            self.Y_pred = tf.reshape(Y_pred, [self.batch_size, 1, self.output_size])
            #self.Y_pred =  self.Y_pred[:,-1,:] # batch, sequence_length, output_size -> batch, 1, output_size

    def build_loss(self, lr, lr_decay_rate, lr_decay_step):
        self.init_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        batch_size = self.batch_size

        with tf.variable_scope('lstm_loss'):
            # loss = tf.losses.mean_squared_error(Y, outputs, weights=weights)#reduction=tf.losses.Reduction.MEAN)
            # loss = tf.reduce_mean(tf.square(Y-outputs))
            self.loss = tf.reduce_sum(tf.square(self.Y_data[:, -1, :] - self.Y_pred))
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
