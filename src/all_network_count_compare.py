# IOU Calculation
import numpy as np
import matplotlib.pyplot as plt
import random
from patchify import unpatchify
import os
import tifffile as tiff
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Predict and calculate Intersection over Union (IoU) for different thresholds
y_pred = model.predict(X_test)  # Predict the output for the test dataset
IOUs = []  # List to store IoU scores
dices = []  # List to store Dice coefficients
thresh = []  # List to store threshold values

# Iterate over thresholds from 0.0 to 1.0 in steps of 0.1
for threshold in np.linspace(0, 1, 11):
    y_pred_thresholded = y_pred > threshold  # Apply threshold to predictions
    intersection = np.logical_and(y_test, y_pred_thresholded)  # Calculate intersection between predicted and ground truth masks
    union = np.logical_or(y_test, y_pred_thresholded)  # Calculate union between predicted and ground truth masks
    # Add check to avoid division by zero
    if np.sum(union) == 0:
        iou_score = 0.0
    else:
        iou_score = np.sum(intersection) / np.sum(union)  # Compute IoU score
    # Calculate Dice coefficient with a check to avoid division by zero
    dice_c = (2 * np.sum(intersection)) / (np.sum(y_test) + np.sum(y_pred_thresholded)) if (np.sum(y_test) + np.sum(y_pred_thresholded)) > 0 else 0
    print("IoU score is: ", round(iou_score, 4), "Dice coeff is: ", round(dice_c, 4), "at threshold: ", threshold)
    IOUs.append(iou_score)
    thresh.append(threshold)
    dices.append(dice_c)

# Plot IoU and Dice coefficient vs. threshold
plt.plot(thresh, IOUs, 'r', label='IOU')
plt.plot(thresh, dices, 'b', label='Dice coeff')
plt.title('IOU & Dice vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('IOU')
plt.legend()
plt.show()

#######################################################################
# Predict on a few images
idx = np.random.choice(len(X_test))  # Randomly select an index from the test set

plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.title('Testing Image')
plt.imshow(X_test[idx, 1, :, :])  # Display the test image
plt.subplot(132)
plt.title('Testing Label')
plt.imshow(y_test[idx, 0, :, :])  # Display the corresponding ground truth label
plt.subplot(133)
plt.title('Prediction on Test Image')
plt.imshow(y_pred[idx, 0, :, :])  # Display the predicted mask
plt.tight_layout()
plt.show()

# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
acc = history.history.get('dice_coef', [0] * len(epochs))  # Placeholder for Dice coefficient
val_acc = history.history.get('val_dice_coef', [0] * len(epochs))
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#######################################################################
# Define helper functions for unpatching predictions

def _createOutArr(shape, nrows, ncols, nchannels):
    # Calculate the output array dimensions based on the patch shape and number of rows/columns
    out_height = int(nrows * shape[-2])
    out_width = int(ncols * shape[-1])
    out_frames = int(shape[0] / (nrows * ncols))
    outshape = (out_frames, nchannels, out_height, out_width)
    out_arr = np.zeros(outshape, dtype=np.float32)  # Use np.zeros to avoid uninitialized memory
    return out_arr

def unpatcher(arr, nrows, ncols, nchannels=1):
    # Reconstruct the original image from patches
    out_arr = _createOutArr(arr.shape, nrows, ncols, nchannels)
    patch_h = arr.shape[-2]  # Height of each patch
    patch_w = arr.shape[-1]  # Width of each patch
    n = 0
    for frame in range(out_arr.shape[0]):
        for i in range(nrows):
            for j in range(ncols):
                y = patch_h * i  # Calculate the y-coordinate for placing the patch
                x = patch_w * j  # Calculate the x-coordinate for placing the patch
                out_arr[frame, :, y:y+patch_h, x:x+patch_w] = arr[n]  # Place the patch in the correct position
                n += 1
    return out_arr

#######################################################################
# Load unseen data and predict
validation_image_directory = r"C:\Users\analyst\Documents\Python Scripts\BactUnet\Bactnet\Training data\stacks\predict"
result_folder = r"C:\Users\analyst\Documents\Python Scripts\BactUnet\_results"

# Load images from the validation directory
images = [img for img in os.listdir(validation_image_directory) if img.split('.')[-1].lower() == 'tif']  # Filter non-empty stacks

for i, image_name in enumerate(images):
    image = tiff.imread(os.path.join(validation_image_directory, image_name))  # Load the image using tifffile
    original_shape = image.shape  # Save the original shape for later use
    patch = patch_stack(image, SIZE)  # Break the image into patches
    patch = normalizePercentile(patch, 0.1, 99.9, clip=True)  # Normalize the patches using percentile-based normalization
    pred_mask_patch = model.predict(patch, batch_size=16)  # Predict the mask for each patch with parameterized batch size
    print(image_name, original_shape, patch.shape, pred_mask_patch.shape)
    image = np.expand_dims(patch[:, 1, :, :], axis=1)  # Expand dimensions for concatenation
    patch = np.concatenate((image, pred_mask_patch), axis=1)  # Concatenate the original and predicted masks
    unpatched = unpatcher(patch, 8, 8, 2)  # Reconstruct the original image from patches
    # Save the unpatched result as a TIFF file
    tiff.imwrite(os.path.join(result_folder, image_name), unpatched, imagej=True, resolution=(1./2.6755, 1./2.6755),
                  metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})

# Let's try the full movies

image_directory = r"C:\Users\analyst\Documents\Python Scripts\BactUnet\Bactnet"
result_folder = r"C:\Users\analyst\Documents\Python Scripts\BactUnet\_results"
filelist = []

for dir in os.listdir(image_directory):
    for file in os.listdir(os.path.join(image_directory, dir)):
        if ".tif" in file:
            loadme = os.path.join(image_directory, dir, file)
            image = tiff.imread(loadme)
            original_shape = image.shape
            patch = patch_stack(image, SIZE)
            patchlist = np.array_split(patch, 238)  # Use np.array_split for better performance and clarity
            
            for i, p in enumerate(patchlist):
                p = normalizePercentile(p, 0.1, 99.9, clip=True)
                pred_mask_patch = model.predict(p, batch_size=16)  # Predict with parameterized batch size
                unpatched = unpatcher(pred_mask_patch, 8, 8, 1)
                savename = file.split(".")[0]+ "_V3_" + str(i+1) + ".tif"
                print(savename, unpatched.shape)
                tiff.imwrite(os.path.join(result_folder, savename), unpatched, imagej=True, resolution=(1./2.6755, 1./2.6755),
                             metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})