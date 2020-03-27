import tensorflow as tf
from tensorflow.keras.layers import Layer
import memory_retention_vector, allocation_weighting, sorting_indices, usage_vector, write_weight_interpolation
from models import content_lookup

def dynamic_memory(free_gates, read_weighting_prev, prev_usage_vector, prev_write_weighting, allocation_gate, write_gate, memory_matrix, key, key_strength):
    retention_vector = memory_retention_vector.memory_retention_vector(free_gates, read_weighting_prev)
    usage_vector = usage_vector.usage_vector(retention_vector, prev_usage_vector, prev_write_weighting)
    sorted_usage_vector = sorting_indices.sorting_indices(usage_vector)
    allocation_vector = allocation_weighting.allocation_weighting(sorted_usage_vector)
    content_vector = content_lookup(memory_matrix, key, key_strength)
    write_weighing = write_weight_interpolation.write_weight_interpolation(allocation_gate, write_gate, allocation_vector, content_vector)
    return write_weighting, usage_vector
