import tensorflow as tf

def temporal_gates(temporal_links, write_weights):
  forward_gate = tf.matmul(write_weights, 
      temporal_links, transpose_b = True)
  backward_gate = tf.matmul(write_weights, temporal_links)
  return forward_gate, backward_gate
