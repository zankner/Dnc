import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose,
    BatchNormalization, MaxPool2D, Flatten, Dot
)
from tensorflow.keras import Model
from models.temporal_link import temporal_link
from models.read_heads import get_read_vectors
from models.write_heads import write


class Network(Model):
  def __init__(self, memory_locations, memory_slot_size, batch_size, input_dim, output_dim, interface_dim, read_heads):
    self.flatten = Flatten()
    self.d1 = Dense(input_dim, activation='relu')
    self.weights_output = Dense(output_dim) #previously n
    self.weights_read = Dense(output_dim)
    self.weights_xi = Dense(memory_slot_size * read_heads + 3 * memory_slot_size + 5 * read_heads + 3) #previously g
    self.prev_usage_vector = tf.convert_to_tensor(np.random.random((memory_locations, 1)))
    self.read_weighting_prev = tf.convert_to_tensor(np.random.random((memory_locations, 1)))
    self.prev_write_vector = tf.convert_to_tensor(np.random.random((memory_locations, 1)))
    self.memory_matrix = tf.convert_to_tensor(np.random.random((memory_locations, memory_slot_size)))


  def call(self, input, read_weighting, training=False):

    #Here fetch the read read_vectors through read_weighting
    read_vectors = get_read_vectors(self.memory_matrix, read_weighting)
    #feed forward
    read_vectors = self.flatten(read_vectors)
    input_vector = tf.concat(input, read_vectors)
    network_output = self.d1(input_vector)

    #controller output vector
    v = self.weights_output(network_output)
    output_vector = tf.math.add(v, self.weights_read(read_vectors))

    #interface vector
    interface_vector = self.weights_xi(h)
    r_keys, r_strengths, write_key, write_strength, \
        e, w, free_gates, allocation_gate, write_gate, r_modes = tf.split(
            interface_vector, [memory_slot_size*read_heads, read_heads, memory_slot_size, 1, memory_slot_size, memory_slot_size, read_heads, 1, 1, read_heads*3], 1)
    r_gates = [tf.split(r_gates, [r for r in range(read_heads)])]


    #Accesing memory
    write_vector, usage_vector = dynamic_memory(free_gates, self.read_weighting_prev, self.prev_usage_vector, self.prev_write_vector,
                                                        allocation_gate, write_gate, self.memory_matrix, write_key, write_strength)
    temporal_links, precedence = temporal_link(temporal_links, precedence, write_weightings)

    #update stored variables
    self.prev_usage_vector = usage_vector
    self.read_weighting_prev = read_weighting
    self.prev_write_vector = write_vector

    return output_vector, interface_vector
