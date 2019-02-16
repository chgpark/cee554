import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 5
num_anchor = 4

cell1 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

# x_data = np.array([[[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
#                    [[2,2,2,2],[0,0,0,0],[0,0,4,0],[0,1,2,3]],
#                    [[3,2,1,0],[0,5,6,7],[0,1,2,3],[0,4,4,4]]],dtype = np.float32)

x_data = np.array([[[1,1,1],[0,0,0],[0,0,0],[0,0,0]],
                   [[2,2,2],[0,0,0],[0,4,0],[1,2,3]],
                   [[3,2,1],[0,5,6],[0,2,3],[4,4,4]]],dtype = np.float32)

x_data = np.array([[[1,1],[0,0],[0,0],[0,0]]],dtype = np.float32)


x_data_multimodal= np.array([[[1],[0],[0],[0]],[[2],[0],[0],[0]], [[3],[0],[0],[0]]],dtype = np.float32)

test_object = 'bi'
if test_object == 'uni':
    outputs, _state = tf.nn.dynamic_rnn(cell1, x_data, dtype = tf.float32)
if test_object == 'uni_multimodal':
    outputs, _state = tf.nn.dynamic_rnn(cell1, x_data_multimodal, dtype = tf.float32)
elif test_object =='bi':
    outputs, _state = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, x_data_multimodal, dtype = tf.float32)

sess.run(tf.global_variables_initializer())

total_num_parameters = 0
for variable in tf.trainable_variables():
    total_num_parameters += np.array(variable.get_shape().as_list()).prod()
print("number of trainable parameters: {}".format(total_num_parameters))
