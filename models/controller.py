import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose,
    BatchNormalization, MaxPool2D, Flatten, Dot
)
from tensorflow.keras import Model
from models.temporal_link import temporal_link
from models.read_vectors import read_vectors
from models.write import write
from models.read_weightings import read_weightings


class Network(Model):
  def __init__(self, N, M, net_dim, output_dim, R):
    self.flatten = Flatten()
    self.d1 = Dense(net_dim, activation='relu')
    self.weights_output = Dense(output_dim) #previously n
    self.weights_read = Dense(output_dim)
    self.weights_xi = Dense((N * R) + (3 * N) + (5 * read_heads) + 3) 

  def call(self, x, usage, r_weighting, w_weighting, w_weighting,
      w, precedence, memory_matrix, temporal_links,training=False):

    #feed forward
    r = self.flatten(r)
    x = tf.concat(x, r)
    network_output = self.d1(x)

    #interface vector
    interface_vector = self.weights_xi(network_output)
    r_keys, r_strengths, w_key, w_strengths, e, w, f_gates, a_gate, \
        w_gate, r_modes = tf.split(interface_vector, 
            [N*R, R, N, 1, N, N, R, 1, 1, 3*R, 1)
    r_gates = [tf.split(r_gates, [r for r in range(R)])]
    r_strengths = one_plus(r_strengths)
    w_strengths = one_plus(w_strengths)
    e = tf.math.log_sigmoid(e)
    f_gates = tf.math.log_sigmoid(f_gates)
    a_gate = tf.math.log_sigmoid(a_gate)
    w_gate = tf.math.log_sigmoid(w_gate)
    r_modes = tf.nn.softmax(r_modes)

    #Dynamic memory------ Generating the write_weighting
    w_weighting, usage = dynamic_memory(f_gates, r_weighting, usage, 
      w_weighting, a_gate, w_gate, memory_matrix, w_key, w_strength)

    #Temporal Links and Generating the Read weighting
    temporal_links, precedence = temporal_link(temporal_links, precedence,
      w_weightings)
    forward_weight, backward_weight = temporal_weights(temporal_links, 
      r_weighting)
    r_weighting = read_weighting(r_modes, backward_weighting, 
        content_weighting, forward_weighting)

    # Write to Memory
    memory_matrix = write(memory_matrix, w_weighting, e, w)

    #controller output vector
    v = self.weights_output(network_output)
    r = read_vector(memory_matrix, r_weighting)
    output_vector = tf.math.add(v, self.weights_read(r))

    return output_vector, interface_vector, usage, r_weighting, \
        w_weighting, w, precedence, memory_matrix, temporal_links
