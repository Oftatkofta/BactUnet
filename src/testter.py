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

source_path = r'Bactnet\Training data\stacks\validation_source'
validation_mask_directory = r'Bactnet\Training data\stacks\validation_target'

val_image_dataset = []
pred_mask_dataset = []

stacks = os.listdir(source_path)

SIZE=288

for i, image_name in enumerate(stacks):  # Enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        image = tiff.imread(os.path.join(source_path, image_name))
        image = normalizePercentile(image, 0.1, 99.9, clip=True)
        expanded_image = pad_stack(image, SIZE)
        patch = patch_stack(image, SIZE)
        expanded_patch = patch_stack(expanded_image, SIZE)
        patch2 = patch_stack(expanded_image, SIZE)
        up_exp_img = unpatch_stack(patch2[:, 1, :,:], 9, 9, 1)
        crop = crop_stack(up_exp_img, SIZE)

        print(image_name, image.shape, patch.shape, expanded_image.shape, expanded_patch.shape, up_exp_img.shape, crop.shape)




plt.imshow(crop[1, 1, : ,:])
plt.show()

