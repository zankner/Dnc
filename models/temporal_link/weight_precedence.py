import tensorflow as tf

def weight_precedence(precedence, write_weight):
  precedence_weighting = 1 - tf.reduce_sum(
      write_weight, 1, keepdims = True)
  weighted_precedence = tf.math.multiply(
      precedence_weighting, precedence)
  precedence = weighted_precedence + write_weight
  return precedence
