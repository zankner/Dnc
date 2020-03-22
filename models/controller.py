import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose,
    BatchNormalization, MaxPool2D, Flatten, Dot
)
from tensorflow.keras import Model
from models.temporal_link import temporal_link


class Network(Model):

    #output dimensionality
    n = 10

    #interface dimensionality
    g = 10

  def __init__(self):
    super(UNet, self).__init__()
    self.flatten = Flatten()
    self.d1 = Dense(100, activation='relu')
    self.weights_output = Dense(n)
    self.weights_read = Dense(n)
    self.weights_xi = Dense(g)


  def call(self, input, read_vectors, training=False):

    #feed forward
    read_vectors = self.flatten(read_vectors)
    input_vector = tf.concat(input, read_vectors)
    network_output = self.d1(input_vector)

    #controller output vector
    v = self.weights_output(network_output)
    output_vector = tf.math.add(v, self.weights_read(read_vectors))

    #interface vector
    interface_vector = self.weights_xi(h)
    r_keys, r_strenghts, w_key, w_strength, \
        e, w, r_gates, allocation, w_gate, r_models = tf.split(
            interface_vector, [W*R, R, W, 1, W, W, R, 1, 1, R*3], 1)
    r_gates = [tf.split(r_gates, [r for r in range(R)])]

    #Accesing memory
    temporal_links, precedence = temporal_link(temporal_links,
        precedence, write_weightings)

    return output_vector, interface_vector
