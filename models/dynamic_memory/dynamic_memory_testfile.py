import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import allocation_weighting, sorting_indices, usage_vector, write_weight_interpolation

test_allocation_weighting = tf.convert_to_tensor(np.array([[2.3, 10, 3],[2.4, 12, 12], [2.5, 24,2], [2.6, 30,3]])) # should be nxc vector.
#in this case, the size is 4x3
test_result1 = allocation_weighting.allocation_weighting(test_allocation_weighting)
print("test_allocation " + str(test_result1.get_shape()))
#test_result1 shape should be 4x3

test_sorting = sorting_indices.sorting_indices(test_allocation_weighting)
print("test_sorting" + str(test_sorting.get_shape()))
#test_sorting should return 4x3

memory_retention_vector = tf.convert_to_tensor(np.array([[1, 2.2, 4.3, 4, 1.2], [4, 2.4, 2.3, 5, 1.2]])) #1xN
prev_usage_vector = tf.convert_to_tensor(np.transpose(np.array([[1, 2.2, 120, 4, 1.2], [3, 5, 7, 5.43, 1.22]]))) #Nx1
prev_write_vector = tf.convert_to_tensor(np.transpose((np.array([[1, 2.5, 3.3, 4, 1.2], [10, 2.4, 13, 4, 1.2]])))) #Nx1

test_usage_vector = usage_vector.usage_vector(memory_retention_vector, prev_usage_vector, prev_write_vector)
print(test_usage_vector)
intermediate_test = sorting_indices.sorting_indices(test_usage_vector)
test_result2 = allocation_weighting.allocation_weighting(intermediate_test)
print(test_result2)

interpolation_test = write_weight_interpolation.write_weight_interpolation(0.4, 0.2, test_result2, tf.convert_to_tensor(np.transpose(np.array([[1.227,2.43,5,3,22],[1,2.51,3.998,6,3]]))))
print(interpolation_test)
