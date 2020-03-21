import tensorflow as tf

def temporal_weights(temporal_links, write_weights):
  forward_weights = tf.matmul(write_weights, 
      temporal_links, transpose_b = True)
  backward_weights = tf.matmul(write_weights, temporal_links)
  return forward_weights, backward_weights
