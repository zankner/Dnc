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
    self.weights_chi = Dense(g)

  def call(self, x, r, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training

    #feed forward
    r = self.flatten(r)
    z = tf.concat(x, r)
    h = self.d1(z)

    #controller output vector
    v = self.weights_output(h)
    y = tf.math.add(v, self.weights_read(r))

    #interface vector
    chi = self.weights_chi(h)

    return y, chi
