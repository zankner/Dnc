import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose,
    BatchNormalization, MaxPool2D, Flatten, Dot
)
from tensorflow.keras import Model

#Defining network Below:
class Network(Model):

    #output dimensionality
    n = 10

    #interface dimensionality
    g = 10

  def __init__(self):
    super(UNet, self).__init__()
    # Define layers of the network:
    self.flatten = Flatten()
    self.d1 = Dense(100, activation='relu')
    self.weights_output = Dense(n)
    self.weights_read = Dense(n)
    self.weights_xi = Dense(g)

  def call(self, input, read_vectors, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training

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
    return output_vector, interface_vector
