import tensorflow as tf

def write(m, w, e, v):

    e = tf.transpose(e)
    v = tf.transpose(v)
    big_e = tf.ones(tf.shape(m).numpy())

    m = tf.math.multiply(m, tf.math.add(tf.math.subtract(big_e, tf.math.multiply(w, e)), tf.math.multiply(w, v)))
    return m
