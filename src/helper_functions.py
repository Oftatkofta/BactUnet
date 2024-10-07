# helper_functions.py

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from tensorflow.keras import backend as K

    def calculate_layer_memory(layer, batch_size):
        """
        Calculate the memory usage for a specific layer.
        
        Parameters:
        layer: The layer for which memory usage is being calculated.
        batch_size: The batch size used in model training.
        
        Returns:
        shapes_mem_count: The memory required for storing layer outputs.
        internal_model_mem_count: The memory used by nested models, if any.
        """
        shapes_mem_count = 0
        internal_model_mem_count = 0
        layer_type = layer.__class__.__name__
        # If the layer is a nested model, calculate memory usage recursively
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, layer)
        else:
            single_layer_mem = 1
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            # Calculate the product of dimensions to determine memory usage
            for dim in out_shape:
                if dim is None:
                    continue
                single_layer_mem *= dim
            shapes_mem_count += single_layer_mem
        return shapes_mem_count, internal_model_mem_count

    total_shapes_mem_count = 0
    total_internal_model_mem_count = 0
    # Iterate through each layer of the model to calculate memory usage
    for layer in model.layers:
        shapes_mem_count, internal_mem_count = calculate_layer_memory(layer, batch_size)
        total_shapes_mem_count += shapes_mem_count
        total_internal_model_mem_count += internal_mem_count

    # Count the number of trainable and non-trainable parameters
    trainable_count = K.count_params(model.trainable_weights)
    non_trainable_count = K.count_params(model.non_trainable_weights)

    # Determine the size of each number based on the floating-point precision
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    elif K.floatx() == 'float64':
        number_size = 8.0

    # Calculate total memory usage in bytes and convert to gigabytes
    total_memory = number_size * (batch_size * total_shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + total_internal_model_mem_count
    return gbytes