import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 2
num_anchor = 4

cell1 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

x_data1 = np.array([[[1,0,0,0]]],dtype = np.float32)
x_data2 = np.array([[[1,1,0,0]]],dtype = np.float32)
x_data3 = [x_data1, x_data2]
outputs_list = []
outputs, _state = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, x_data1, dtype = tf.float32)
outputs_list.append(outputs)


outputs2, _state = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, x_data2, dtype = tf.float32)
outputs3, _state = tf.nn.bidirectional_dynamic_rnn(cell1, cell2, x_data3, dtype = tf.float32)
outputs_list.append(outputs2)
# outputs, _state = tf.nn.dynamic_rnn(cell1, x_data, dtype = tf.float32)


sess.run(tf.global_variables_initializer())

print(outputs_list)
#Because outputs is tuple, so we need to convert to tuple
outputs_list = tf.convert_to_tensor(outputs_list)
outputs_list2 = tf.convert_to_tensor(outputs3)
# print(outputs.shape)
a = np.array([1,2,3])
b = np.array([5,2,3])
output2 = tf.add(a,b)
# print((outputs))
pp.pprint(outputs_list.eval())
pp.pprint(outputs_list2.eval())

# print(outputs)
