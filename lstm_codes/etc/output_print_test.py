import tensorflow as tf
import pprint
import numpy as np
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

hidden_size = 5
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
b = tf.one_hot(indices, depth, on_value = 1.0, off_value = 0.0, axis = -1)



#
print (b.eval())


# def get_vector(sequence_input):
#     a = []
#     batch_size_of_seq = sequence_input.shape[0]
#     sequence_size_of_seq = sequence_input.shape[1]
#     for i in range(batch_size_of_seq):
#         sequence = sequence_input[i]
#         vector = []
#         for j in range(sequence_size_of_seq - 1):
#             v = sequence[j + 1] - sequence[j]
#             vector.append(v.tolist())
#         a.append(vector)
#
#     return np.array(a)
#
# a= get_vector(x_data1)
# b= get_vector(x_data2)
# print (a)
# # print (tf.reduce_sum(tf.square(a)).eval())
# # print (np.linalg.norm([1,2]))
# norm_a =  np.linalg.norm(a, axis = 2)+0.00000001
# norm_b =  np.linalg.norm(b, axis = 2)+0.00000001


# d = tf.constant(10, dtype= tf.float64)
# print (a*b)
# print (tf.reduce_mean(1 - (tf.reduce_sum(a*b, axis = 2)/(norm_a*norm_b))).eval())
# c = tf.add(a,d)
# print (c.eval())
# print (a*b/(norm_a*norm_b))
# print (np.dot(a,b)/(norm_a* norm_b))
# print (np.dot([1,2],[2,3])/np.linalg.norm([1,2]))
# outputs2 = tf.convert_to_tensor(outputs2)

# pp.pprint(outputs2.eval())
# print ("concatenate")

