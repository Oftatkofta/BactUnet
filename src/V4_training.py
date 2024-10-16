import tensorflow as tf
import os
import tifffile as tiff
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import logging

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout, MaxPool2D, Concatenate
from keras_unet_collection import losses

# Import helper functions from preprocessing.py, window_functions.py, and helper_functions.py
from preprocessing import patch_image, patch_stack, normalizePercentile, normalize_mi_ma, normalizeMinMax, checkEmptyMask
from window_functions import hanning_window, hamming_window, blackman_window, kaiser_window, bartlett_window, apply_window
from helper_functions import get_model_memory_usage

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

pmin = 0.1
pmax = 99.9

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print('You do not have GPU access.')
else:
    print('You have GPU access')
    for gpu in gpus:
        print(f"GPU: {gpu.name}")

# Print TensorFlow and Keras versions
print(f"TensorFlow: {tf.__version__}; Keras: {keras.__version__}")

# Display GPU device name if available
gpu_device_name = tf.test.gpu_device_name()
if gpu_device_name:
    print(f"GPU Device: {gpu_device_name}")
else:
    print("No GPU device found.")

train_path = r"Bactnet/Training data/stacks"

batch_size = 16
SIZE = 288

def prepare_data(train_path, PATCH_SIZE, delete_empty=False, validation=False, seam=False):
    if validation:
        prefix = "validation"
    else:
        prefix = "training"
    
    stacks = [stack for stack in os.listdir(os.path.join(train_path, prefix + "_source")) if stack.endswith(".tif")]
    image_dataset = []
    mask_dataset = []
    for stack in stacks:
        img = tiff.imread(os.path.join(train_path, prefix + "_source", stack))
        mask = tiff.imread(os.path.join(train_path, prefix + "_target", stack))
        
        # Patch the images and masks based on whether seam patches are needed
        if seam:  # should the patches be created over the seams of the "standard"?
            half = PATCH_SIZE // 2
            img = patch_stack(img[1:-1, half:-half, half:-half], PATCH_SIZE, DEPTH=1)
            mask = patch_stack(mask[:, half:-half, half:-half], PATCH_SIZE, DEPTH=1)
        else:
            img = patch_stack(img[1:-1], PATCH_SIZE, DEPTH=1)
            mask = patch_stack(mask, PATCH_SIZE, DEPTH=1)
        
        print(stack, img.shape, mask.shape)
        # Normalize the mask and image
        mask = normalizeMinMax(mask)
        img = normalizePercentile(img, pmin, pmax, clip=True)
        
        # Delete empty patches if specified
        if delete_empty:
            not_ok_idxs = checkEmptyMask(mask)
            mask = np.delete(mask, not_ok_idxs, axis=0)
            img = np.delete(img, not_ok_idxs, axis=0)
            print(stack, img.shape, mask.shape)

        image_dataset.append(img)
        mask_dataset.append(mask)

    if image_dataset:
        image_dataset = np.concatenate(image_dataset)
    else:
        image_dataset = np.array([])

    if mask_dataset:
        mask_dataset = np.concatenate(mask_dataset)
    else:
        mask_dataset = np.array([])

    return image_dataset, mask_dataset

# Prepare the training and testing datasets
image_dataset, mask_dataset = prepare_data(train_path, SIZE, delete_empty=True, validation=False, seam=False)
seam_data = prepare_data(train_path, SIZE, delete_empty=True, validation=False, seam=False)
X_train = np.concatenate((image_dataset, seam_data[0]))
y_train = np.concatenate((mask_dataset, seam_data[1]))

X_test, y_test = prepare_data(train_path, SIZE, delete_empty=True, validation=True, seam=False)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Sanity check, view a few images
import random

image_number = random.randint(0, X_train.shape[0])
print(image_number)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number, 0, :, :])
plt.subplot(122)
plt.imshow(y_train[image_number, 0, :, :])
plt.show()

def conv_block(input, num_filters):
    # Convolutional block with two Conv2D layers followed by BatchNormalization and Dropout
    t = Conv2D(num_filters, 3, padding="same", data_format="channels_first", activation='relu')(input)
    t = BatchNormalization()(t)
    t = Dropout(0.1)(t)
    t = Conv2D(num_filters, 3, padding="same", data_format="channels_first", activation='relu')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.1)(t)
    return t

# Encoder block: Conv block followed by max pooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2), data_format="channels_first")(x)
    return x, p

# Decoder block: Upsampling followed by concatenation and conv block
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", data_format="channels_first")(input)
    x = Concatenate(axis=1)([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# Build Unet using the encoder and decoder blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder path
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder path
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output layer with a sigmoid activation for binary segmentation
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid", data_format="channels_first")(d4)

    model = Model(inputs, outputs, name="BactUnet_single_frame_training")
    return model

if keras.backend.is_keras_tensor(tf.constant(0)):  # Determine if clearing the session is necessary
    keras.backend.clear_session()  # Free up RAM in case the model definition cells were run multiple times

#########################################################

# Build model
def hybrid_loss(y_true, y_pred):

    alpha = 0.3  # Parameter to control the balance of the loss
    gamma = 4 / 3  # Parameter to control the focusing level of the loss

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=alpha, gamma=gamma)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal + loss_iou #+loss_ssim

SIZE=288
input_shape = (1, SIZE, SIZE)
batch_size = 8

model = build_unet(input_shape)
model.compile(loss=hybrid_loss,
            #loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=[losses.dice_coef, losses.iou_seg])
model.summary()
print(get_model_memory_usage(batch_size, model))

# ##################################
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

# New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
# This gives a binary mask rather than a mask with interpolated values. 
seed = 1337

# Data augmentation parameters for images and masks
img_data_gen_args = dict(rotation_range=180,
                     width_shift_range=0.25,
                     height_shift_range=0.25,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     data_format="channels_first")

mask_data_gen_args = dict(rotation_range=180,
                     width_shift_range=0.25,
                     height_shift_range=0.25,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     data_format="channels_first",
                     preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype))  # Binarize the output again.

# Image and mask data generators for training and validation
image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(X_test, seed=seed, batch_size=batch_size)  # Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed, batch_size=batch_size)  # Default batch size 32, if not specified here

# Custom generator that yields image and mask pairs
def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

# Create generators for training and validation
my_generator = my_image_mask_generator(image_generator, mask_generator)
validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

# Function to visualize data generated by the generators
def visualize_generator_data(generator, title):
    x, y = generator.next()
    print(x.shape, y.shape)
    for i in range(len(x)):
        image = x[i] if title == "Image" else y[i]
        fig1 = plt.subplot(4, 4, i + 1)
        fig1.set_xticks([])  # Turn off axis
        fig1.set_yticks([])
        plt.imshow(image[0, :, :], cmap='gray')
    plt.show()

# Visualize a batch of images and masks
visualize_generator_data(valid_img_generator, "Image")
visualize_generator_data(valid_mask_generator, "Mask")

# Calculate steps per epoch based on training data size
steps_per_epoch = 3 * (len(X_train)) // batch_size

# Function to create callback list for model training
def create_callbacks(model_name, filepath_suffix="_logs.csv"):
    filepath = r"models/" + model_name + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    log_csv = CSVLogger(r'models/' + model_name + filepath_suffix, separator=',', append=False)
    return [checkpoint, early_stop, log_csv]

# Initial training parameters
epochs = 100
model_name = "bactunet_V4_single_frame"
callbacks_list = create_callbacks(model_name)

# Train the model
history = model.fit(
    my_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_datagen,
    validation_steps=steps_per_epoch,
    callbacks=callbacks_list)

# Save the trained model
model.save(r"models/" + model_name + ".hdf5")

# Continue training with additional epochs
callbacks_list = create_callbacks(model_name="bactnet_v3", filepath_suffix="_logs.csv")

history = model.fit(
    my_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=1000,
    validation_data=validation_datagen,
    validation_steps=steps_per_epoch,
    callbacks=callbacks_list)

# Save the final trained model
model.save(r"models/bactunet_noEmpty_final_alldata.hdf5")

# ##################################
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
    dice_c = (2 * np.sum(intersection)) / (np.sum(y_test) + np.sum(y_pred_thresholded)) if (np.sum(y_test) + np.sum(y_pred_thresholded)) > 0 else 0  # Calculate Dice coefficient
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
    pred_mask_patch = model.predict(patch)  # Predict the mask for each patch
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
                pred_mask_patch = model.predict(p)
                unpatched = unpatcher(pred_mask_patch, 8, 8, 1)
                savename = file.split(".")[0]+ "_V3_" + str(i+1) + ".tif"
                print(savename, unpatched.shape)
                tiff.imwrite(os.path.join(result_folder, savename), unpatched, imagej=True, resolution=(1./2.6755, 1./2.6755),
                             metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})



    