import tensorflow as tf
import numpy as np

def allocation_weighting(ordered_usage_vector):
    ability_to_allocate = (tf.cast(tf.ones(tf.shape(ordered_usage_vector)), tf.float64) - ordered_usage_vector)
    righthand_usage_value = tf.math.reduce_prod(ordered_usage_vector, axis=1)
    righthand_usage_value = tf.transpose(tf.expand_dims(righthand_usage_value, axis=0))
    #elems = (ability_to_allocate, righthand_usage_value)
    #allocation_vector = tf.map_fn(lambda x, y: x * y, elems[0], elems[1], dtype=tf.float64)
    allocation_vector = tf.multiply(righthand_usage_value, ability_to_allocate)
    return allocation_vector
