import tensorflow as tf
import numpy as np

def allocation_weighting(ordered_usage_vector):
    ability_to_allocate = (tf.ones(tf.shape(ordered_usage_vector).numpy()) - ordered_usage_vector)
    righthand_usage_value = tf.math.reduce_prod(ordered_usage_vector)
    allocation_vector = ability_to_allocate * rigthand_usage_value
    return allocation_vector
