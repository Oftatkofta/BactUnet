import tensorflow as tf
from patchify import patchify, unpatchify
from skimage import img_as_ubyte, io, transform
from matplotlib import pyplot as plt
import os
import tifffile as tiff
import numpy as np
import math
from matplotlib import pyplot as plt
from preprocessing import *


PATCH_SIZE = 288




# load unseen data

validation_image_directory = r'Bactnet\Training data\stacks\validation_source'
validation_mask_directory = r'Bactnet\Training data\stacks\validation_target'

val_image_dataset = []
pred_mask_dataset = []

images = os.listdir(validation_image_directory)
SIZE=288
pad_SIZE = int(SIZE/2)

def pad_image_arr(arr, SIZE):
    """
    Zero-pads arr with 1/2 SIZE in x and y so that when patched the patches are
    centered on the seams of the patches from the unpadded image
    """
    pad_SIZE = int(SIZE / 2)
    expanded_image = np.pad(image_dataset, ((0, 0), (pad_SIZE, pad_SIZE), (pad_SIZE, pad_SIZE)))

    return expanded_image
for i, image_name in enumerate(images[0:1]):  # Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        image = tiff.imread(os.path.join(validation_image_directory, image_name))
        image = normalizePercentile(image, 0.1, 99.9, clip=True)
        original_shape = image.shape
        patch = patch_stack(image, SIZE)
        expanded_image = np.pad(image, ((0,0), (pad_SIZE, pad_SIZE), (pad_SIZE, pad_SIZE)))
        expanded_patch = patch_stack(expanded_image, SIZE)
        #patch = normalizePercentile(patch, 0.1, 99.9, clip=True)
        #pred_mask_patch = patch[:,1,:,:].reshape((640,1,288,288))
        #pred_mask_patch = np.concatenate([pred_mask_patch] * 3, axis=1)
        print(image_name, original_shape, patch.shape, expanded_image.shape, expanded_patch.shape)
        #pred_mask = unpatch_stack(pred_mask_patch, original_shape)
        #print(pred_mask.shape)
        # tiff.imsave(os.path.join(result_folder, image_name), pred_mask)
        # val_image_dataset.append(image)
        # pred_mask_dataset.append(pred_mask)


plt.imshow(expanded_patch[1, 1, : ,:])
plt.show()

