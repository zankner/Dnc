import tensorflow as tf

def content_lookup(memory_matrix, key, key_strength):

    def cosine_sim(u, v):
        return tf.math.truediv(tf.linalg.matmul(u, v), tf.math.multiply(tf.norm(u), tf.norm(v)))

    key = tf.transpose(key)

    return tf.nn.softmax(cosine_sim(memory_matrix, key) * key_strength)
