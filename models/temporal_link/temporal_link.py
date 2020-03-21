import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose, 
    BatchNormalization, concatenate, MaxPool2D
)
from tensorflow.keras.layers import Layer

#Defining temporal linking Below:
class TemporalLinking(Layer):
  def __init__(self):
    super(TemporalLinking, self).__init__()
    # Define layers of the network:

  def call(self, x, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    precedence = weight_precedence(precedence, write_weight)

    return x
