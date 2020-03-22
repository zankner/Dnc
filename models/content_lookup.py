import tensorflow as tf

def content_lookup(memory_matrix, key, key_strength):

    def cosine_sim(u, v):
        return tf.math.divide(tf.linalg.mathmul(u, v), tf.linalg.mathmul(tf.math.abs(u), tf.math.abs(v))

    key = tf.transpose(key)
    return tf.nn.softmax(cosine_sim(key, memory_matrix) * key_strength)
