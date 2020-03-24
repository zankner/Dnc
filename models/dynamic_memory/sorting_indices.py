import tensorflow as tf
import numpy as np

def sorting_indices(usage_vector):
    free_list_unsorted = usage_vector #Nx1 Matrix
    unsorted_memory_usage = usage_vector

    print(usage_vector)

    free_list_sorted = tf.argsort(free_list_unsorted, axis = 0, direction = 'ASCENDING', stable=True)
    free_list_sorted_index = tf.keras.backend.eval(free_list_sorted)
    #free_list_sorted_index = tf.transpose(free_list_sorted_index)
    #print(free_list_sorted_index)
    ordered_memory_usage = tf.gather(unsorted_memory_usage, free_list_sorted_index, axis=0)
    #print(ordered_memory_usage)
    ordered_memory_usage = tf.linalg.diag_part(ordered_memory_usage)
    #step to fetch the eye of the Matrix


    return ordered_memory_usage
