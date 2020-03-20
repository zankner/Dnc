import tensorflow
import numpy as np

def memory_retention_vector(forget_gate_list, read_weighting_prev, read_head_count):
    output_list = []
    assert len(forget_gate_list) == read_head_count
    for read_headn in forget_gate_list:
        read_weighting_slices = tf.gather(read_weighting_prev, [read_headn])
        #output_element = 1 - (forget_gate * read_weighting_prev[:][read_headn])
        output_element = tf.ones(tf.shape(read_weighting_slices).numpy()) - (tf.math.scalar_mul(forget_gate, read_weighting_slices))
        output_list.append(output_element)

    #output_vector = np.array(output_list)
    output_vector = tf.convert_to_tensor(output_list)
    #output_vector = np.prod(output_vector, axis = 0)
    #output vector should be 1 x N. One row and N locations(columns).
    output_vector = tf.math.reduce_prod(output_vector)
    return output_vector
