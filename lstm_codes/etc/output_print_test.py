import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 2
num_anchor = 4



def getAttentionedOutput(tensor):
    attention = tf.nn.sigmoid(tensor)
    attentioned_tensor = attention*tensor
    return attentioned_tensor + tensor

cell1 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

cell3 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
cell4 = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

x_data1 = np.array([[[1],[0],[0],[0]],[[2],[0],[0],[0]], [[3],[0],[0],[0]]],dtype = np.float32)
print (x_data1.shape)
x_data2 = np.array([[1,1,0,0]],dtype = np.float32)
x_data3 = [x_data1, x_data2]
outputs_list = []
with tf.variable_scope("1"):
    outputs, _state = tf.nn.dynamic_rnn(cell1, x_data1, dtype = tf.float32)
with tf.variable_scope("22"):
    outputs2, _state2 = tf.nn.dynamic_rnn(cell3, x_data1, dtype = tf.float32)

# outputs_2, _state_2 = tf.nn.dynamic_rnn(cell1, x_data1, dtype = tf.float32)
sess.run(tf.global_variables_initializer())

outputs = tf.convert_to_tensor(outputs)
outputs2 = tf.convert_to_tensor(outputs2)
pp.pprint(outputs.eval())
pp.pprint(outputs2.eval())
concat = tf.concat([outputs, outputs2], axis = 2 )
# print ("concatenate")
pp.pprint(concat.eval())

# pp.pprint(outputs[0].eval())
# print (outputs.shape)
# print (outputs.shape[1])
# print (outputs[0].shape)
# print (outputs[0].shape[0])
# array = []
# for i in range(outputs.shape[1]):
#     a = tf.concat([outputs[0][i], outputs[1][i]], axis = 1)
#     array.append(a)
# array = tf.convert_to_tensor(array)
# pp.pprint(array.eval())
# array2 = getAttentionedOutput(array)
# pp.pprint(array2.eval())
