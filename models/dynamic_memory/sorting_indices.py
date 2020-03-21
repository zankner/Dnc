import tensorflow as tf
import numpy as np

def sorting_indices(usage_vector, memory_locations):
    free_list_unsorted = usage_vector
    unsorted_memory_locations = memory_locations #N*W matrix

    free_list_sorted = tf.argsort(free_list_unsorted, axis = -1, direction = 'ASCENDING', stable=False)
    free_list_sorted_index = tf.keras.backend.eval(free_list_sorted)

    ordered_memory_locations = tf.gather(unsorted_memory_locations, free_list_sorted_index)

    return ordered_memory_locations
