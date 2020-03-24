import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import allocation_weighting, sorting_indices

test_allocation_weighting = tf.convert_to_tensor(np.array([[2.3, 10, 3],[2.4, 12, 12], [2.5, 24,2], [2.6, 30,3]])) # should be nxc vector.
#in this case, the size is 4x3
test_result1 = allocation_weighting.allocation_weighting(test_allocation_weighting)
print("test_allocation " + str(test_result1.get_shape()))
#test_result1 shape should be 4x3

test_sorting = sorting_indices.sorting_indices(test_allocation_weighting)
print("test_sorting" + str(test_sorting.get_shape()))
#test_sorting should return 4x3
