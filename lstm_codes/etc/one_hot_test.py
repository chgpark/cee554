import tensorflow as tf
indices = [[1,2,],[0,1]]

depth = 3
b = tf.one_hot(indices, depth, on_value = 1.0, off_value = 0.0, axis = -1)

print (b)