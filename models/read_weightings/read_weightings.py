import tensorflow as tf

def read_weightings(read_mode, backward_weighting, 
    forward_weighting, content_weighting):
  backward_mode, content_mode, forward_mode = tf.split(
      read_mode, [1,1,1], 2)
  backward_mode = tf.reshape(backward_mode, 
      [backward_mode.shape[0], backward_mode.shape[1]])
  content_mode = tf.reshape(content_mode, 
      [content_mode.shape[0], content_mode.shape[1]])
  forward_mode = tf.reshape(forward_mode, 
      [forward_mode.shape[0], forward_mode.shape[1]])
  backward_weighting = tf.transpose(backward_weighting)
  backward_mode = tf.transpose(backward_mode)
  backward_component = backward_mode * backward_weighting
  backward_component = tf.transpose(backward_component)
  content_weighting = tf.transpose(content_weighting)
  content_mode = tf.transpose(content_mode)
  content_component = content_mode * content_weighting
  content_component = tf.transpose(content_component)
  forward_weighting = tf.transpose(forward_weighting)
  forward_mode = tf.transpose(forward_mode)
  forward_component = forward_mode * forward_weighting
  forward_component = tf.transpose(forward_component)
  return backward_component + content_component + forward_component
