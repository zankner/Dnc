import tensorflow as tf

def read_vectors(memory_matrix, read_weightings):
    return tf.linalg.matmul(tf.transpose(memory_matrix), read_weightings)
