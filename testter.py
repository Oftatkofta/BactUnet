import tensorflow as tf
from patchify import patchify, unpatchify
from skimage import img_as_ubyte, io, transform
from matplotlib import pyplot as plt
import os
import tifffile as tiff
import numpy as np
import math
from matplotlib import pyplot as plt
#from simple_unet_model import get_model, DataGen, get_model_memory_usage


def patch_image(img, SIZE=288):
    # breaks up image to SIZExSIZE non-overlapping patches, returns reshaped array
    # img.shape = (2304, 2304)
    patches = patchify(img, patch_size=(SIZE, SIZE), step=(SIZE, SIZE))
    # patches.shape = (8, 8, 288, 288)

    # return.shape = (64, 288 ,288)
    return patches.reshape(patches.shape[0] * patches.shape[1], SIZE, SIZE)

def patch_stack(img, SIZE=288, DEPTH=3, STRIDE=1):
    # breaks up image to SIZExSIZE non-overlapping patches, returns reshaped array
    # img.shape = (12, 2304, 2304)
    patches = patchify(img, patch_size=(DEPTH, SIZE, SIZE), step=(STRIDE, SIZE, SIZE))
    # patches.shape = (10, 8, 8, 3, 288, 288)

    # return.shape = (64, 3, 288 ,288)
    return patches.reshape(patches.shape[0] * patches.shape[1] * patches.shape[2], -1,  SIZE, SIZE)



# Normalization functions from Martin Weigert
def normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):  # dtype=np.float32
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


# Simple normalization to min/max fir the Mask
def normalizeMinMax(x, dtype=np.float32):
    x = x.astype(dtype, copy=False)
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    return x


def unpatch_image(patches, SIZE, original_shape):
    side_length = int(math.sqrt(patches.shape[0]))
    p = patches.reshape(side_length, side_length, SIZE, SIZE)

    return unpatchify(p, original_shape)

def unpatch_stack(patches, original_shape, DEPTH=3):

    n_frames = original_shape[0]-(DEPTH-1)
    side_length = int(math.sqrt(patches.shape[0]/n_frames))
    patches = patches.reshape(n_frames, side_length, side_length, DEPTH, patches.shape[2] , -1)

    return unpatchify(patches, original_shape)

def bin_frames_in_three(img):
    #Bins frames into 3-frame chunks with stride 1
    #Assumes img.shape = (frames, x, y)
    #returns (frames-2, 3, x, y)
    out = np.zeros((img.shape[0]-2, 3, img.shape[1], img.shape[2]), dtype=np.float32)
    for frame in range(0, img.shape[0]-2):
        out[frame] = img[frame:frame+3]
    return out

def checkEmptyMask(arr):
    #checks if any patches are without masks
    #returns list of indexes where mask is all zeros
    out = []
    for i in range(arr.shape[0]):
        if not arr[i].any():
            out.append(i)

    return out


def prepare_data(train_path, PATCH_SIZE):
    stacks = os.listdir(os.path.join(train_path, "training_source"))
    image_dataset = None
    mask_dataset = None
    for stack in stacks:
        if (stack.split(".")[-1]=="tif"):
            img = tiff.imread(os.path.join(train_path, "training_source",stack))
            mask = tiff.imread(os.path.join(train_path, "training_target", stack))

            img = patch_stack(img, PATCH_SIZE)
            mask =patch_stack(mask, PATCH_SIZE)
            mask = mask[:, 1, :, :]

            print(stack, img.shape, mask.shape)
            mask = normalizeMinMax(mask)
            not_ok_idxs = checkEmptyMask(mask)
            mask = np.delete(mask, not_ok_idxs, axis=0)
            img = np.delete(img, not_ok_idxs, axis=0)

            img = normalizePercentile(img, 0.1, 99.9, clip=True)

            print(stack, img.shape, mask.shape)

            if image_dataset is not None:
                image_dataset = np.concatenate((image_dataset, img))

            if mask_dataset is not None:
                mask_dataset = np.concatenate((mask_dataset, mask))

            if image_dataset is None:
                image_dataset = img

            if mask_dataset is None:
                mask_dataset = mask

            print(image_dataset.shape, mask_dataset.shape)

    return image_dataset, mask_dataset

train_path = r"C:\Users\Jens\Documents\Code\bactnet\Bactnet\Training data\stacks"
PATCH_SIZE = 288


def unpatch_stack(patches, original_shape, DEPTH=3):
    n_frames = original_shape[0] - (DEPTH - 1)
    side_length = int(math.sqrt(patches.shape[0] / n_frames))
    patches = patches.reshape(n_frames, side_length, side_length, DEPTH, patches.shape[2], -1)

    return unpatchify(patches, original_shape)


# load unseen data

validation_image_directory = r'C:\Users\Jens\Documents\Code\bactnet\Bactnet\Training data\stacks\predict'

val_image_dataset = []
pred_mask_dataset = []

images = os.listdir(validation_image_directory)
SIZE=288
for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        image = tiff.imread(os.path.join(validation_image_directory, image_name))
        original_shape = image.shape
        patch = patch_stack(image, SIZE)
        #patch = normalizePercentile(patch, 0.1, 99.9, clip=True)
        pred_mask_patch = patch[:,1,:,:].reshape((640,1,288,288))
        pred_mask_patch = np.concatenate([pred_mask_patch] * 3, axis=1)
        print(image_name, original_shape, patch.shape, pred_mask_patch.shape)
        pred_mask = unpatch_stack(pred_mask_patch, original_shape)
        print(pred_mask.shape)
        # tiff.imsave(os.path.join(result_folder, image_name), pred_mask)
        # val_image_dataset.append(image)
        # pred_mask_dataset.append(pred_mask)
plt.imshow(pred_mask[4])
plt.show()

