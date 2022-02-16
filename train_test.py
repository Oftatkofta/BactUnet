# https://youtu.be/csFGTLT6_WQ
"""
Author: Dr. Sreenivas Bhattiprolu
Training and testing for semantic segmentation (Unet) of mitochondria
Uses standard Unet framework with no tricks!
Dataset info: Electron microscopy (EM) dataset from
https://www.epfl.ch/labs/cvlab/data/data-em/
Patches of 256x256 from images and labels
have been extracted (via separate program) and saved to disk.
This code uses 256x256 images/masks.
To annotate images and generate labels, you can use APEER (for free):
www.apeer.com
"""

from simple_unet_model import get_model, DataGen   #Use normal unet model
#from keras.utils import normalize
import os
import tifffile as tiff
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from tensorflow import keras

def patch_image(img, SIZE=288):
    #breaks up image to SIZExSIZE non-overlapping patches, returns reshaped array
    # img.shape = (2304, 2304)
    patches = patchify(img, patch_size=(SIZE, SIZE), step=(SIZE, SIZE))
    #patches.shape = (8, 8, 288, 288)

    #return.shape = (64, 288 ,288)
    return patches.reshape(patches.shape[0]*patches.shape[1], SIZE, SIZE, 1)

def unpatch(patches, SIZE=288):
    p = patches.reshape




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

def checkEmptyMask(arr):
    #checks if any patches are without masks
    #returns list of indexes where mask is all zeros
    out = []
    for i in range(arr.shape[0]):
        if not arr[i].any():
            out.append(i)

    return out

image_directory = r"C:\Users\Jens\Desktop\Bactnet\Training data\all\training_source"
mask_directory = r"C:\Users\Jens\Desktop\Bactnet\Training data\all\training_target"

batch_size = 5
SIZE = 144
image_dataset = None
mask_dataset = None

images = os.listdir(image_directory)

for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):

        mask = tiff.imread(os.path.join(mask_directory, image_name))
        mask = patch_image(mask, SIZE)
        mask = normalizeMinMax(mask)
        not_ok_idxs = checkEmptyMask(mask)
        mask = np.delete(mask, not_ok_idxs, axis=0)

        if mask_dataset is not None:
            mask_dataset = np.concatenate((mask_dataset, mask))

        if mask_dataset is None:
            mask_dataset = mask

        image = tiff.imread(os.path.join(image_directory, image_name))
        image = patch_image(image, SIZE)
        image = normalizePercentile(image, 0.1, 99.9, clip=True)
        image = np.delete(image, not_ok_idxs, axis=0)

        if image_dataset is not None:
            image_dataset = np.concatenate((image_dataset, image))

        if image_dataset is None:
            image_dataset = image



print(image_dataset.shape, mask_dataset.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Sanity check, view few mages
import random

image_number = random.randint(0, X_train.shape[0])
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number])
plt.subplot(122)
plt.imshow(y_train[image_number])
plt.show()

###############################################################


keras.backend.clear_session() # Free up RAM in case the model definition cells were run multiple times

# Build model
model = get_model((SIZE, SIZE), 1)
model.summary()
train_gen = DataGen(batch_size, X_train, y_train)
test_gen = DataGen(batch_size, X_test, y_test)

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="adam", loss="binary_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("bactnet_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
#model.fit(train_gen, epochs=epochs, validation_data=test_gen, callbacks=callbacks)


# #If starting with pre-trained weights.
model.load_weights("bactnet_segmentation.h5")
#
# history = model.fit(X_train, y_train,
#                      batch_size = 16,
#                      verbose=1,
#                      epochs=1,
#                      validation_data=(X_test, y_test),
#                      shuffle=False)

# model.save('bactnet_test.hdf5')
#
# ############################################################
#Evaluate the model


# 	# evaluate model
# _, acc = model.evaluate(test_gen)
# print("Accuracy = ", (acc * 100.0), "%")
#
#
# #plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# acc = history.history['acc']
# #acc = history.history['accuracy']
# val_acc = history.history['val_acc']
# #val_acc = history.history['val_accuracy']
#
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# ##################################
# #IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#Predict on a few images
#model = get_model()
#model.load_weights('mitochondria_50_plus_100_epochs.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

idx=2

plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.title('Testing Image')
plt.imshow(X_test[idx])
plt.subplot(132)
plt.title('Testing Label')
plt.imshow(y_test[idx])
plt.subplot(133)
plt.title('Prediction on test image')
plt.imshow(y_pred[idx])

plt.show()

# #plt.imsave('input.jpg', test_img[:,:,0], cmap='gray')
# #plt.imsave('data/results/output2.jpg', prediction_other, cmap='gray')
#
#

