import tensorflow as tf
import numpy as np

def mem_retention(free_gates, read_weightings):
  return tf.linalg.matvec(free_gates, read_weightings, transpose_a=True)
