import tensorflow as tf

def write(memory_matrix, write_weighting, erase_vector, write_vector):

    erase_vector = tf.transpose(erase_vector)
    write_vector = tf.transpose(write_vector)
<<<<<<< HEAD
<<<<<<< HEAD
    big_e = tf.ones(tf.shape(memory_matrix).numpy())
=======
    big_e = tf.ones(tf.shape(m).numpy())
>>>>>>> dd1f766acd75c005d31d31010694c81fe3dfd103
=======
    big_e = tf.ones(tf.shape(m).numpy())
>>>>>>> 8bcc09e93155063792910161cb8f49261791b036

    memory_matrix = tf.math.multiply(memory_matrix, tf.math.add(tf.math.subtract(big_e, tf.linalg.matmul(write_weighting, erase_vector)), tf.linalg.matmul(write_weighting, write_vector)))
    return memory_matrix
