import tensorflow
import numpy as np

def usage_vector(memory_retention_vector, prev_usage_vector, prev_write_vector):
    assert prev_write_vector[:][0].shape == prev_usage_vector[0][:].shape #the write vector should be the
    #same size as the usage vector in that dimension.
    vector_before_retention = (prev_usage_vector + prev_write_vector - np.multiply(prev_usage_vector, prev_write_vector))
    memory_retention_vector = np.transpose(memory_retention_vector)
    final_vector = np.multiply(vector_before_retention, memory_retention_vector)
    return final_vector
    #notes: essentially, the usage vector should always return a size of (n*1),
    #which is why the memory retention vector is always transposed, since its initially
    #it is of size (1*n).
