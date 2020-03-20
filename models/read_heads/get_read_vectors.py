import tensorflow as tf

def get_read_vectors(m, w):
    return tf.math.multiply(tf.transpose(m), w)
    
