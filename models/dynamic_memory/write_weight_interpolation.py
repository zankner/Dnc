import tensorflow as tf
import numpy as np

def write_weight_interpolation(allocation_gate, write_gate, allocation_vector, content_vector):
    allocation_term = allocation_gate * allocation_vector
    content_term = (tf.ones(tf.shape(allocagtion_gate).numpy()) - allocation_gate) * content_vector
    write_information = allocation_term + content_term
    write_weight = write_gate * write_information
    return write_weight
