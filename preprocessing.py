import numpy as np
import math
from patchify import patchify, unpatchify
from tensorflow import keras


def patch_image(img, SIZE=288):
    # breaks up image to SIZExSIZE non-overlapping patches, returns reshaped array
    # img.shape = (2304, 2304)
    patches = patchify(img, patch_size=(SIZE, SIZE), step=(SIZE, SIZE))
    # patches.shape = (8, 8, 288, 288)
    # return.shape = (64, 1, 288 ,288)
    return patches.reshape(patches.shape[0] * patches.shape[1], -1, SIZE, SIZE)


def patch_stack(img, SIZE=288, DEPTH=3, STRIDE=1):
    # breaks up image to SIZExSIZE non-overlapping patches, returns reshaped array
    # img.shape = (12, 2304, 2304)
    patches = patchify(img, patch_size=(DEPTH, SIZE, SIZE), step=(STRIDE, SIZE, SIZE))
    # patches.shape = (10, 8, 8, 3, 288, 288)

    # return.shape = (64, 3, 288 ,288)
    return patches.reshape(patches.shape[0] * patches.shape[1] * patches.shape[2], -1, SIZE, SIZE)


def _unpatch_stack(patches, original_shape, DEPTH=3):
    #DELETE LATER
    n_frames = original_shape[0] - (DEPTH - 1)
    side_length = int(math.sqrt(patches.shape[0] / n_frames))
    patches = patches.reshape(n_frames, side_length, side_length, DEPTH, patches.shape[2], -1)

    return unpatchify(patches, original_shape)


def pad_stack(arr, SIZE):
    """
    Zero-pads arr with 1/2 SIZE in x and y so that when patched the patches are
    centered on the seams of the patches from the unpadded image
    """

    pad_SIZE = int(SIZE / 2)
    expanded_image = np.pad(arr, ((0, 0), (pad_SIZE, pad_SIZE), (pad_SIZE, pad_SIZE)))

    return expanded_image


def crop_stack(arr, SIZE):
    """
    Undoes the padding from pad_stack, crops out the center, removing a 1/2 SIZE broder from around the stack
    """
    pad_SIZE = int(SIZE / 2)

    if len(arr.shape) == 3:
        t, y, x = arr.shape
        stopY = y - pad_SIZE
        stopX = x - pad_SIZE
        return arr[:, pad_SIZE:stopY, pad_SIZE:stopX]

    if len(arr.shape) == 4:
        t, c, y, x = arr.shape
        stopY = y - pad_SIZE
        stopX = x - pad_SIZE
        return arr[:, :, pad_SIZE:stopY, pad_SIZE:stopX]

def normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=True, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=True, eps=1e-20, dtype=np.float32):  # dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def normalizeMinMax(x, dtype=np.float32):
    # Simple normalization to min/max for the Mask
    x = x.astype(dtype, copy=False)
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    return x


def checkEmptyMask(arr):
    """
    Checks if any patches are without masks, previously used to discard patches with no mask.
    returns list of indexes where mask is all zeros
    """

    out = []
    for i in range(arr.shape[0]):
        if not arr[i].any():
            out.append(i)

    return out


def unpatch_stack(arr, nrows, ncols, nchannels=1):
    """
    Undoes what patch_stack does.
    Takes an numpy array of patches and stitches them back together.
    """

    out_arr = _createOutArr(arr.shape, nrows, ncols, nchannels)
    patch_h = arr.shape[-2]
    patch_w = arr.shape[-1]
    n = 0
    for frame in range(out_arr.shape[0]):
        for i in range(nrows):
            for j in range(ncols):
                y = patch_h * i
                x = patch_w * j
                out_arr[frame, :, y:y + patch_h, x:x + patch_w] = arr[n]
                n += 1

    return out_arr

def _createOutArr(shape, nrows, ncols, nchannels):
    out_height = int(nrows * shape[-2])
    out_width = int(ncols * shape[-1])
    out_frames = int(shape[0] / (nrows * ncols))

    outshape = (out_frames, nchannels, out_height, out_width)

    out_arr = np.empty(outshape, dtype=np.float32)

    return out_arr


def bin_frames_in_three(img):
    # Bins frames into 3-frame chunks with stride 1
    # Assumes img.shape = (frames, x, y)
    # returns (frames-2, 3, x, y)
    out = np.zeros((img.shape[0] - 2, 3, img.shape[1], img.shape[2]), dtype=np.float32)
    for frame in range(0, img.shape[0] - 2):
        out[frame] = img[frame:frame + 3]
    return out



def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def predict_stack(arr, batch_size, model):
    """
    Performs prediction on all images in arr using model in increments of batch_size
    Assumes patches of a ahpe where N is 0th axis.
    """
    keras.backend.clear_session()
    y_pred = None
    for i in range(0, len(arr), batch_size):
        pred = model.predict(arr[i:i + batch_size])
        if y_pred is not None:
            y_pred = np.concatenate((y_pred, pred))

        else:
            y_pred = pred

    return y_pred
