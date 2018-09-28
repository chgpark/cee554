import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 2

cell1 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

x_data = np.array([[[1,0,0,0]]],dtype = np.float32)

outputs, _state = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, x_data, dtype = tf.float32)
# outputs, _state = tf.nn.dynamic_rnn(cell1, x_data, dtype = tf.float32)

sess.run(tf.global_variables_initializer())

#Because outputs is tuple, so we need to convert to tuple
outputs= tf.convert_to_tensor(outputs)
print(outputs.shape)

# print((outputs))
pp.pprint(outputs.eval())
# print(outputs)
