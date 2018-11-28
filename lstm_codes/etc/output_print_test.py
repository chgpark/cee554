import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 3
num_anchor = 4
seq_length = 4


def getAttentionedOutput(tensor):
    attention = tf.nn.sigmoid(tensor)
    attentioned_tensor = attention*tensor
    return attentioned_tensor + tensor

cell1 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

cell3 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell4 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

x_data1 = np.array([[[1, 2],[0,5],[0,7],[3,-2]],[[2,1],[0,1],[0,1],[0,1]]],dtype = np.float32)
x_data2 = np.array([[[1, 2],[0,5],[0,7],[3,-2]],[[2,1],[0,1],[0,1],[0,1]]],dtype = np.float32)
print (x_data1.shape)
x_data3 = [x_data1, x_data2]
outputs_list = []
with tf.variable_scope("1"):
    outputs, _state = tf.nn.dynamic_rnn(cell1, x_data1, dtype = tf.float32)
with tf.variable_scope("22"):
    outputs2, _state2 = tf.nn.dynamic_rnn(cell3, x_data1, dtype = tf.float32)

# outputs_2, _state_2 = tf.nn.dynamic_rnn(cell1, x_data1, dtype = tf.float32)
indices = [[1,2,],[0,1]]

depth = 3

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (outputs.eval())
    round_value0 = tf.round(tf.divide(outputs[:,:,0], 0.001))
    round_value1 = tf.round(tf.divide(outputs[:,:,1], 0.001))
    round_value2 = tf.round(tf.divide(outputs[:,:,2], 0.001))
    round_value0 = tf.reshape(round_value0, [2, 4, 1])
    round_value1 = tf.reshape(round_value1, [2, 4, 1])
    round_value2 = tf.reshape(round_value2, [2, 4, 1])

    round_val = tf.concat((round_value0*0.001, round_value1*0.001, round_value2*0.001), axis = 2)
    print (round_val.eval())
    #
    # print (outputs.shape)
# #

