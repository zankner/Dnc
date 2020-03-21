import tensorflow as tf
from tensorflow.keras.layers import Layer

#Defining temporal linking Below:
def temporal_link(tempral_links, precedence, write_weightings):
  temporal_links = update_temporal_links(
      temporal_links, write_weightings, precedence)
  precedence = weight_precedence(precedence, write_weightings)
  return temporal_links, precedence 
