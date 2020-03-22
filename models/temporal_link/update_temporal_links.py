import tensorflow as tf

def update_temporal_links(temporal_links, write_weights, precedence):
  summed_write_weights = tf.math.reduce_sum(write_weights, 0)
  write_weight_j = tf.broadcast_to(
      summed_write_weights, [write_weights.shape[1], 
      write_weights.shape[1]])
  write_weight_i = tf.transpose(write_weight_j)
  temporal_links_weighting = 1 - write_weight_i - write_weight_j
  weighted_temporal_links = tf.math.multiply(
      temporal_links, temporal_links_weighting)

  summed_precedence = tf.math.reduce_sum(precedence, 0)
  precedence_expanded = tf.transpose(tf.broadcast_to(
      summed_precedence, [precedence.shape[1], precedence.shape[1]]))
  weighted_precedence = tf.math.multiply(
      write_weight_i, precedence_expanded)
  temporal_links = weighted_temporal_links + weighted_precedence

  temporal_idenity = tf.eye(temporal_links.shape[0])
  intra_temporal_mask = tf.cast(
      tf.math.equal(temporal_idenity, 0), tf.float32)
  temporal_links = tf.math.multiply(temporal_links, intra_temporal_mask)
  return temporal_links
