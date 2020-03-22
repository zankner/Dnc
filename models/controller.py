import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose,
    BatchNormalization, MaxPool2D, Flatten, Dot
)
from tensorflow.keras import Model


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

    return output_vector, interface_vector
