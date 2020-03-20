import tensorflow as tf
import numpy as np

def sorting_indices(usage_vector, memory_locations):
    free_list_keys = usage_vector
    free_list_values = memory_locations

    hashtable = tf.lookup.StaticHashTable((tf.lookup.KeyValueTensorInitializer(free_list_keys, free_list_values), -1))

    init_index = tf.sort(usage_vector, axis = -1, direction = 'ASCENDING', name = None)
    free_list_keys_sorted = tf.keras.backend.eval(init_index)

    #have to add later code for tf.scan through the free_list of keys and seeing where they back, and pick the index append
    #the new indices into a seperate array according. Should be O(n) operation.
    hashtable.lookup(current_tensor)

    return free_list

    #Program plan, first set the usage vector and memory_locations into a hash table. Then,
    #sort the free_list_keys. After the keys are sorted, then compare back with hash table.

    #tf.argsort ===> actually for sorting I should have used arg sort, then I need no hashtable
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
    #tf.argsort
