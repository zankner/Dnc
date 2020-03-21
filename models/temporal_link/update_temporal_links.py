import tensorflow as tf

def update_temporal_links(temporal_links, write_weights, precedence):
  summed_write_weights = tf.math.reduce_sum(write_weights, 0)
  write_weight_j = tf.broadcast_to(
      summed_write_weights, [write_weights.shape[1], 
      write_weights.shape[1]])
  print(write_weight_j)
  write_weight_i = tf.transpose(write_weight_j)
  print(write_weight_i)

update_temporal_links(3, tf.random.uniform([10, 5]), 3)
