Notebook_version = '1.13'
Network = 'U-Net (2D)'

from builtins import any as b_any


def get_requirements_path():
    # Store requirements file in 'contents' directory 
    current_dir = os.getcwd()
    dir_count = current_dir.count('/') - 1
    path = '../' * (dir_count) + 'requirements.txt'
    return path


def filter_files(file_list, filter_list):
    filtered_list = []
    for fname in file_list:
        if b_any(fname.split('==')[0] in s for s in filter_list):
            filtered_list.append(fname)
    return filtered_list


def build_requirements_file(before, after):
    path = get_requirements_path()

    # Exporting requirements.txt for local run
    !pip
    freeze > $path

    # Get minimum requirements file
    df = pd.read_csv(path, delimiter="\n")
    mod_list = [m.split('.')[0] for m in after if not m in before]
    req_list_temp = df.values.tolist()
    req_list = [x[0] for x in req_list_temp]

    # Replace with package name and handle cases where import name is different to module name
    mod_name_list = [['sklearn', 'scikit-learn'], ['skimage', 'scikit-image']]
    mod_replace_list = [[x[1] for x in mod_name_list] if s in [x[0] for x in mod_name_list] else s for s in mod_list]
    filtered_list = filter_files(req_list, mod_replace_list)

    file = open(path, 'w')
    for item in filtered_list:
        file.writelines(item + '\n')

    file.close()


import sys

before = [str(m) for m in sys.modules]

# @markdown ##Load key U-Net dependencies

# As this notebokk depends mostly on keras which runs a tensorflow backend (which in turn is pre-installed in colab)
# only the data library needs to be additionally installed.
% tensorflow_version
1.
x
import tensorflow as tf
# print(tensorflow.__version__)
# print("Tensorflow enabled.")


# Keras imports
from keras import models
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger # we currently don't use any other callbacks from ModelCheckpoints
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as keras

# General import

import numpy as np
import pandas as pd
import os
import glob
from skimage import img_as_ubyte, io, transform
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from pathlib import Path
import shutil
import random
import time
import csv
import sys
from math import ceil
from fpdf import FPDF, HTMLMixin
from pip._internal.operations.freeze import freeze
import subprocess
# Imports for QC
from PIL import Image
from scipy import signal
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr

# For sliders and dropdown menu and progress bar
from ipywidgets import interact
import ipywidgets as widgets
# from tqdm import tqdm
from tqdm.notebook import tqdm

from sklearn.feature_extraction import image
from skimage import img_as_ubyte, io, transform
from skimage.util.shape import view_as_windows

from datetime import datetime

# Suppressing some warnings
import warnings

warnings.filterwarnings('ignore')


def create_patches(Training_source, Training_target, patch_width, patch_height, min_fraction):
    """
    Function creates patches from the Training_source and Training_target images.
    The steps parameter indicates the offset between patches and, if integer, is the same in x and y.
    Saves all created patches in two new directories in the /content folder.

    min_fraction is the minimum fraction of pixels that need to be foreground to be considered as a valid patch

    Returns: - Two paths to where the patches are now saved
    """
    DEBUG = False

    Patch_source = os.path.join('/content', 'img_patches')
    Patch_target = os.path.join('/content', 'mask_patches')
    Patch_rejected = os.path.join('/content', 'rejected')

    # Here we save the patches, in the /content directory as they will not usually be needed after training
    if os.path.exists(Patch_source):
        shutil.rmtree(Patch_source)
    if os.path.exists(Patch_target):
        shutil.rmtree(Patch_target)
    if os.path.exists(Patch_rejected):
        shutil.rmtree(Patch_rejected)

    os.mkdir(Patch_source)
    os.mkdir(Patch_target)
    os.mkdir(Patch_rejected)  # This directory will contain the images that have too little signal.

    patch_num = 0

    for file in tqdm(os.listdir(Training_source)):

        img = io.imread(os.path.join(Training_source, file))
        mask = io.imread(os.path.join(Training_target, file), as_gray=True)

        if DEBUG:
            print(file)
            print(img.dtype)

        # Using view_as_windows with step size equal to the patch size to ensure there is no overlap
        patches_img = view_as_windows(img, (patch_width, patch_height), (patch_width, patch_height))
        patches_mask = view_as_windows(mask, (patch_width, patch_height), (patch_width, patch_height))

        patches_img = patches_img.reshape(patches_img.shape[0] * patches_img.shape[1], patch_width, patch_height)
        patches_mask = patches_mask.reshape(patches_mask.shape[0] * patches_mask.shape[1], patch_width, patch_height)

        if DEBUG:
            print(all_patches_img.shape)
            print(all_patches_img.dtype)

        for i in range(patches_img.shape[0]):
            img_save_path = os.path.join(Patch_source, 'patch_' + str(patch_num) + '.tif')
            mask_save_path = os.path.join(Patch_target, 'patch_' + str(patch_num) + '.tif')
            patch_num += 1

            # if the mask conatins at least 2% of its total number pixels as mask, then go ahead and save the images
            pixel_threshold_array = sorted(patches_mask[i].flatten())
            if pixel_threshold_array[int(round(len(pixel_threshold_array) * (1 - min_fraction)))] > 0:
                io.imsave(img_save_path, img_as_ubyte(normalizeMinMax(patches_img[i])))
                io.imsave(mask_save_path, convert2Mask(normalizeMinMax(patches_mask[i]), 0))
            else:
                io.imsave(Patch_rejected + '/patch_' + str(patch_num) + '_image.tif',
                          img_as_ubyte(normalizeMinMax(patches_img[i])))
                io.imsave(Patch_rejected + '/patch_' + str(patch_num) + '_mask.tif',
                          convert2Mask(normalizeMinMax(patches_mask[i]), 0))

    return Patch_source, Patch_target


def estimatePatchSize(data_path, max_width=512, max_height=512):
    files = os.listdir(data_path)

    # Get the size of the first image found in the folder and initialise the variables to that
    n = 0
    while os.path.isdir(os.path.join(data_path, files[n])):
        n += 1
    (height_min, width_min) = Image.open(os.path.join(data_path, files[n])).size

    # Screen the size of all dataset to find the minimum image size
    for file in files:
        if not os.path.isdir(os.path.join(data_path, file)):
            (height, width) = Image.open(os.path.join(data_path, file)).size
            if width < width_min:
                width_min = width
            if height < height_min:
                height_min = height

    # Find the power of patches that will fit within the smallest dataset
    width_min, height_min = (fittingPowerOfTwo(width_min), fittingPowerOfTwo(height_min))

    # Clip values at maximum permissible values
    if width_min > max_width:
        width_min = max_width

    if height_min > max_height:
        height_min = max_height

    return (width_min, height_min)


def fittingPowerOfTwo(number):
    n = 0
    while 2 ** n <= number:
        n += 1
    return 2 ** (n - 1)


def getClassWeights(Training_target_path):
    Mask_dir_list = os.listdir(Training_target_path)
    number_of_dataset = len(Mask_dir_list)

    class_count = np.zeros(2, dtype=int)
    for i in tqdm(range(number_of_dataset)):
        mask = io.imread(os.path.join(Training_target_path, Mask_dir_list[i]))
        mask = normalizeMinMax(mask)
        class_count[0] += mask.shape[0] * mask.shape[1] - mask.sum()
        class_count[1] += mask.sum()

    n_samples = class_count.sum()
    n_classes = 2

    class_weights = n_samples / (n_classes * class_count)
    return class_weights


def weighted_binary_crossentropy(class_weights):
    def _weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = keras.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
        weighted_binary_crossentropy = weight_vector * binary_crossentropy

        return keras.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy


def save_augment(datagen, orig_img, dir_augmented_data="/content/augment"):
    """
    Saves a subset of the augmented data for visualisation, by default in /content.

    This is adapted from: https://fairyonice.github.io/Learn-about-ImageDataGenerator.html

    """
    try:
        os.mkdir(dir_augmented_data)
    except:
        ## if the preview folder exists, then remove
        ## the contents (pictures) in the folder
        for item in os.listdir(dir_augmented_data):
            os.remove(dir_augmented_data + "/" + item)

        ## convert the original image to array
    x = img_to_array(orig_img)
    ## reshape (Sampke, Nrow, Ncol, 3) 3 = R, G or B
    # print(x.shape)
    x = x.reshape((1,) + x.shape)
    # print(x.shape)
    ## -------------------------- ##
    ## randomly generate pictures
    ## -------------------------- ##
    i = 0
    # We will just save 5 images,
    # but this can be changed, but note the visualisation in 3. currently uses 5.
    Nplot = 5
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dir_augmented_data,
                              save_format='tif',
                              seed=42):
        i += 1
        if i > Nplot - 1:
            break


# Generators
def buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, subset, batch_size,
                         target_size):
    '''
    Can generate image and mask at the same time use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same

    datagen: ImageDataGenerator
    subset: can take either 'training' or 'validation'
    '''
    seed = 1
    image_generator = image_datagen.flow_from_directory(
        os.path.dirname(image_folder_path),
        classes=[os.path.basename(image_folder_path)],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        subset=subset,
        interpolation="bicubic",
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        os.path.dirname(mask_folder_path),
        classes=[os.path.basename(mask_folder_path)],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        subset=subset,
        interpolation="nearest",
        seed=seed)

    this_generator = zip(image_generator, mask_generator)
    for (img, mask) in this_generator:
        # img,mask = adjustData(img,mask)
        yield (img, mask)


def prepareGenerators(image_folder_path, mask_folder_path, datagen_parameters, batch_size=4, target_size=(512, 512)):
    image_datagen = ImageDataGenerator(**datagen_parameters, preprocessing_function=normalizePercentile)
    mask_datagen = ImageDataGenerator(**datagen_parameters, preprocessing_function=normalizeMinMax)

    train_datagen = buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path, 'training',
                                         batch_size, target_size)
    validation_datagen = buildDoubleGenerator(image_datagen, mask_datagen, image_folder_path, mask_folder_path,
                                              'validation', batch_size, target_size)

    return (train_datagen, validation_datagen)


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


# This is code outlines the architecture of U-net. The choice of pooling steps decides the depth of the network. 
def unet(pretrained_weights=None, input_size=(256, 256, 1), pooling_steps=4, learning_rate=1e-4, verbose=True,
         class_weights=np.ones(2)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # Downsampling steps
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

    if pooling_steps > 1:
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

        if pooling_steps > 2:
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)

            if pooling_steps > 3:
                pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
                conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
                conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
                drop5 = Dropout(0.5)(conv5)

                # Upsampling steps
                up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                    UpSampling2D(size=(2, 2))(drop5))
                merge6 = concatenate([drop4, up6], axis=3)
                conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
                conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    if pooling_steps > 2:
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop4))
        if pooling_steps > 3:
            up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    if pooling_steps > 1:
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv3))
        if pooling_steps > 2:
            up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    if pooling_steps == 1:
        up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv2))
    else:
        up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))  # activation = 'relu'

    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)  # activation = 'relu'
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)  # activation = 'relu'
    conv9 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)  # activation = 'relu'
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    # model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['acc'])
    model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_binary_crossentropy(class_weights))

    if verbose:
        model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights);

    return model


def predict_as_tiles(Image_path, model):
    # Read the data in and normalize
    Image_raw = io.imread(Image_path, as_gray=True)
    Image_raw = normalizePercentile(Image_raw)

    # Get the patch size from the input layer of the model
    patch_size = model.layers[0].output_shape[1:3]

    # Pad the image with zeros if any of its dimensions is smaller than the patch size
    if Image_raw.shape[0] < patch_size[0] or Image_raw.shape[1] < patch_size[1]:
        Image = np.zeros((max(Image_raw.shape[0], patch_size[0]), max(Image_raw.shape[1], patch_size[1])))
        Image[0:Image_raw.shape[0], 0: Image_raw.shape[1]] = Image_raw
    else:
        Image = Image_raw

    # Calculate the number of patches in each dimension
    n_patch_in_width = ceil(Image.shape[0] / patch_size[0])
    n_patch_in_height = ceil(Image.shape[1] / patch_size[1])

    prediction = np.zeros(Image.shape)

    for x in range(n_patch_in_width):
        for y in range(n_patch_in_height):
            xi = patch_size[0] * x
            yi = patch_size[1] * y

            # If the patch exceeds the edge of the image shift it back
            if xi + patch_size[0] >= Image.shape[0]:
                xi = Image.shape[0] - patch_size[0]

            if yi + patch_size[1] >= Image.shape[1]:
                yi = Image.shape[1] - patch_size[1]

            # Extract and reshape the patch
            patch = Image[xi:xi + patch_size[0], yi:yi + patch_size[1]]
            patch = np.reshape(patch, patch.shape + (1,))
            patch = np.reshape(patch, (1,) + patch.shape)

            # Get the prediction from the patch and paste it in the prediction in the right place
            predicted_patch = model.predict(patch, batch_size=1)
            prediction[xi:xi + patch_size[0], yi:yi + patch_size[1]] = np.squeeze(predicted_patch)

    return prediction[0:Image_raw.shape[0], 0: Image_raw.shape[1]]


def saveResult(save_path, nparray, source_dir_list, prefix='', threshold=None):
    for (filename, image) in zip(source_dir_list, nparray):
        io.imsave(os.path.join(save_path, prefix + os.path.splitext(filename)[0] + '.tif'),
                  img_as_ubyte(image))  # saving as unsigned 8-bit image

        # For masks, threshold the images and return 8 bit image
        if threshold is not None:
            mask = convert2Mask(image, threshold)
            io.imsave(os.path.join(save_path, prefix + 'mask_' + os.path.splitext(filename)[0] + '.tif'), mask)


def convert2Mask(image, threshold):
    mask = img_as_ubyte(image, force_copy=True)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    return mask


def getIoUvsThreshold(prediction_filepath, groud_truth_filepath):
    prediction = io.imread(prediction_filepath)
    ground_truth_image = img_as_ubyte(io.imread(groud_truth_filepath, as_gray=True), force_copy=True)

    threshold_list = []
    IoU_scores_list = []

    for threshold in range(0, 256):
        # Convert to 8-bit for calculating the IoU
        mask = img_as_ubyte(prediction, force_copy=True)
        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0

        # Intersection over Union metric
        intersection = np.logical_and(ground_truth_image, np.squeeze(mask))
        union = np.logical_or(ground_truth_image, np.squeeze(mask))
        iou_score = np.sum(intersection) / np.sum(union)

        threshold_list.append(threshold)
        IoU_scores_list.append(iou_score)

    return (threshold_list, IoU_scores_list)


# -------------- Other definitions -----------
W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
prediction_prefix = 'Predicted_'

print('-------------------')
print('U-Net and dependencies installed.')


# Colors for the warning messages
class bcolors:
    WARNING = '\033[31m'


# Check if this is the latest version of the notebook

All_notebook_versions = pd.read_csv(
    "https://raw.githubusercontent.com/HenriquesLab/ZeroCostDL4Mic/master/Colab_notebooks/Latest_Notebook_versions.csv",
    dtype=str)
print('Notebook version: ' + Notebook_version)
Latest_Notebook_version = All_notebook_versions[All_notebook_versions["Notebook"] == Network]['Version'].iloc[0]
print('Latest notebook version: ' + Latest_Notebook_version)
if Notebook_version == Latest_Notebook_version:
    print("This notebook is up-to-date.")
else:
    print(
        bcolors.WARNING + "A new version of this notebook has been released. We recommend that you download it at https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki")


def pdf_export(trained=False, augmentation=False, pretrained_model=False):
    class MyFPDF(FPDF, HTMLMixin):
        pass

    pdf = MyFPDF()
    pdf.add_page()
    pdf.set_right_margin(-1)
    pdf.set_font("Arial", size=11, style='B')

    day = datetime.now()
    datetime_str = str(day)[0:10]

    Header = 'Training report for ' + Network + ' model (' + model_name + ')\nDate: ' + datetime_str
    pdf.multi_cell(180, 5, txt=Header, align='L')

    # add another cell
    if trained:
        training_time = "Training time: " + str(hour) + "hour(s) " + str(mins) + "min(s) " + str(round(sec)) + "sec(s)"
        pdf.cell(190, 5, txt=training_time, ln=1, align='L')
    pdf.ln(1)

    Header_2 = 'Information for your materials and method:'
    pdf.cell(190, 5, txt=Header_2, ln=1, align='L')

    all_packages = ''
    for requirement in freeze(local_only=True):
        all_packages = all_packages + requirement + ', '
    # print(all_packages)

    # Main Packages
    main_packages = ''
    version_numbers = []
    for name in ['tensorflow', 'numpy', 'Keras']:
        find_name = all_packages.find(name)
        main_packages = main_packages + all_packages[find_name:all_packages.find(',', find_name)] + ', '
        # Version numbers only here:
        version_numbers.append(all_packages[find_name + len(name) + 2:all_packages.find(',', find_name)])

    cuda_version = subprocess.run('nvcc --version', stdout=subprocess.PIPE, shell=True)
    cuda_version = cuda_version.stdout.decode('utf-8')
    cuda_version = cuda_version[cuda_version.find(', V') + 3:-1]
    gpu_name = subprocess.run('nvidia-smi', stdout=subprocess.PIPE, shell=True)
    gpu_name = gpu_name.stdout.decode('utf-8')
    gpu_name = gpu_name[gpu_name.find('Tesla'):gpu_name.find('Tesla') + 10]
    # print(cuda_version[cuda_version.find(', V')+3:-1])
    # print(gpu_name)
    loss = str(model.loss)[str(model.loss).find('function') + len('function'):str(model.loss).find('.<')]
    shape = io.imread(Training_source + '/' + os.listdir(Training_source)[1]).shape
    dataset_size = len(os.listdir(Training_source))

    text = 'The ' + Network + ' model was trained from scratch for ' + str(number_of_epochs) + ' epochs on ' + str(
        number_of_training_dataset) + ' paired image patches (image dimensions: ' + str(
        shape) + ', patch size: (' + str(patch_width) + ',' + str(patch_height) + ')) with a batch size of ' + str(
        batch_size) + ' and a' + loss + ' loss function,' + ' using the ' + Network + ' ZeroCostDL4Mic notebook (v ' + \
           Notebook_version[
               0] + ') (von Chamier & Laine et al., 2020). Key python packages used include tensorflow (v ' + \
           version_numbers[0] + '), Keras (v ' + version_numbers[2] + '), numpy (v ' + version_numbers[
               1] + '), cuda (v ' + cuda_version + '). The training was accelerated using a ' + gpu_name + 'GPU.'

    if pretrained_model:
        text = 'The ' + Network + ' model was trained for ' + str(number_of_epochs) + ' epochs on ' + str(
            number_of_training_dataset) + ' paired image patches (image dimensions: ' + str(
            shape) + ', patch size: (' + str(patch_width) + ',' + str(patch_height) + ')) with a batch size of ' + str(
            batch_size) + '  and a' + loss + ' loss function,' + ' using the ' + Network + ' ZeroCostDL4Mic notebook (v ' + \
               Notebook_version[
                   0] + ') (von Chamier & Laine et al., 2020). The model was re-trained from a pretrained model. Key python packages used include tensorflow (v ' + \
               version_numbers[0] + '), Keras (v ' + version_numbers[2] + '), numpy (v ' + version_numbers[
                   1] + '), cuda (v ' + cuda_version + '). The training was accelerated using a ' + gpu_name + 'GPU.'

    pdf.set_font('')
    pdf.set_font_size(10.)
    pdf.multi_cell(180, 5, txt=text, align='L')
    pdf.set_font('')
    pdf.set_font('Arial', size=10, style='B')
    pdf.ln(1)
    pdf.cell(28, 5, txt='Augmentation: ', ln=1)
    pdf.set_font('')
    if augmentation:
        aug_text = 'The dataset was augmented by'
        if rotation_range != 0:
            aug_text = aug_text + '\n- rotation'
        if horizontal_flip == True or vertical_flip == True:
            aug_text = aug_text + '\n- flipping'
        if zoom_range != 0:
            aug_text = aug_text + '\n- random zoom magnification'
        if horizontal_shift != 0 or vertical_shift != 0:
            aug_text = aug_text + '\n- shifting'
        if shear_range != 0:
            aug_text = aug_text + '\n- image shearing'
    else:
        aug_text = 'No augmentation was used for training.'
    pdf.multi_cell(190, 5, txt=aug_text, align='L')
    pdf.set_font('Arial', size=11, style='B')
    pdf.ln(1)
    pdf.cell(180, 5, txt='Parameters', align='L', ln=1)
    pdf.set_font('')
    pdf.set_font_size(10.)
    if Use_Default_Advanced_Parameters:
        pdf.cell(200, 5, txt='Default Advanced Parameters were enabled')
    pdf.cell(200, 5, txt='The following parameters were used for training:')
    pdf.ln(1)
    html = """ 
  <table width=40% style="margin-left:0px;">
    <tr>
      <th width = 50% align="left">Parameter</th>
      <th width = 50% align="left">Value</th>
    </tr>
    <tr>
      <td width = 50%>number_of_epochs</td>
      <td width = 50%>{0}</td>
    </tr>
    <tr>
      <td width = 50%>patch_size</td>
      <td width = 50%>{1}</td>
    </tr>
    <tr>
      <td width = 50%>batch_size</td>
      <td width = 50%>{2}</td>
    </tr>
    <tr>
      <td width = 50%>number_of_steps</td>
      <td width = 50%>{3}</td>
    </tr>
    <tr>
      <td width = 50%>percentage_validation</td>
      <td width = 50%>{4}</td>
    </tr>
    <tr>
      <td width = 50%>initial_learning_rate</td>
      <td width = 50%>{5}</td>
    </tr>
    <tr>
      <td width = 50%>pooling_steps</td>
      <td width = 50%>{6}</td>
    </tr>
    <tr>
      <td width = 50%>min_fraction</td>
      <td width = 50%>{7}</td>
  </table>
  """.format(number_of_epochs, str(patch_width) + 'x' + str(patch_height), batch_size, number_of_steps,
             percentage_validation, initial_learning_rate, pooling_steps, min_fraction)
    pdf.write_html(html)

    # pdf.multi_cell(190, 5, txt = text_2, align='L')
    pdf.set_font("Arial", size=11, style='B')
    pdf.ln(1)
    pdf.cell(190, 5, txt='Training Dataset', align='L', ln=1)
    pdf.set_font('')
    pdf.set_font('Arial', size=10, style='B')
    pdf.cell(29, 5, txt='Training_source:', align='L', ln=0)
    pdf.set_font('')
    pdf.multi_cell(170, 5, txt=Training_source, align='L')
    pdf.set_font('')
    pdf.set_font('Arial', size=10, style='B')
    pdf.cell(28, 5, txt='Training_target:', align='L', ln=0)
    pdf.set_font('')
    pdf.multi_cell(170, 5, txt=Training_target, align='L')
    # pdf.cell(190, 5, txt=aug_text, align='L', ln=1)
    pdf.ln(1)
    pdf.set_font('')
    pdf.set_font('Arial', size=10, style='B')
    pdf.cell(21, 5, txt='Model Path:', align='L', ln=0)
    pdf.set_font('')
    pdf.multi_cell(170, 5, txt=model_path + '/' + model_name, align='L')
    pdf.ln(1)
    pdf.cell(60, 5, txt='Example Training pair', ln=1)
    pdf.ln(1)
    exp_size = io.imread('/content/TrainingDataExample_Unet2D.png').shape
    pdf.image('/content/TrainingDataExample_Unet2D.png', x=11, y=None, w=round(exp_size[1] / 8),
              h=round(exp_size[0] / 8))
    pdf.ln(1)
    ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
    pdf.multi_cell(190, 5, txt=ref_1, align='L')
    ref_2 = '- Unet: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.'
    pdf.multi_cell(190, 5, txt=ref_2, align='L')
    # if Use_Data_augmentation:
    #   ref_3 = '- Augmentor: Bloice, Marcus D., Christof Stocker, and Andreas Holzinger. "Augmentor: an image augmentation library for machine learning." arXiv preprint arXiv:1708.04680 (2017).'
    #   pdf.multi_cell(190, 5, txt = ref_3, align='L')
    pdf.ln(3)
    reminder = 'Important:\nRemember to perform the quality control step on all newly trained models\nPlease consider depositing your training dataset on Zenodo'
    pdf.set_font('Arial', size=11, style='B')
    pdf.multi_cell(190, 5, txt=reminder, align='C')

    pdf.output(model_path + '/' + model_name + '/' + model_name + '_training_report.pdf')

    print('------------------------------')
    print('PDF report exported in ' + model_path + '/' + model_name + '/')


def qc_pdf_export():
    class MyFPDF(FPDF, HTMLMixin):
        pass

    pdf = MyFPDF()
    pdf.add_page()
    pdf.set_right_margin(-1)
    pdf.set_font("Arial", size=11, style='B')

    Network = 'Unet 2D'

    day = datetime.now()
    datetime_str = str(day)[0:10]

    Header = 'Quality Control report for ' + Network + ' model (' + QC_model_name + ')\nDate: ' + datetime_str
    pdf.multi_cell(180, 5, txt=Header, align='L')

    all_packages = ''
    for requirement in freeze(local_only=True):
        all_packages = all_packages + requirement + ', '

    pdf.set_font('')
    pdf.set_font('Arial', size=11, style='B')
    pdf.ln(2)
    pdf.cell(190, 5, txt='Loss curves', ln=1, align='L')
    pdf.ln(1)
    exp_size = io.imread(full_QC_model_path + '/Quality Control/QC_example_data.png').shape
    if os.path.exists(full_QC_model_path + '/Quality Control/lossCurvePlots.png'):
        pdf.image(full_QC_model_path + '/Quality Control/lossCurvePlots.png', x=11, y=None, w=round(exp_size[1] / 12),
                  h=round(exp_size[0] / 3))
    else:
        pdf.set_font('')
        pdf.set_font('Arial', size=10)
        pdf.multi_cell(190, 5,
                       txt='If you would like to see the evolution of the loss function during training please play the first cell of the QC section in the notebook.',
                       align='L')
    pdf.ln(2)
    pdf.set_font('')
    pdf.set_font('Arial', size=11, style='B')
    pdf.ln(2)
    pdf.cell(190, 5, txt='Threshold Optimisation', ln=1, align='L')
    # pdf.ln(1)
    exp_size = io.imread(full_QC_model_path + '/Quality Control/' + QC_model_name + '_IoUvsThresholdPlot.png').shape
    pdf.image(full_QC_model_path + '/Quality Control/' + QC_model_name + '_IoUvsThresholdPlot.png', x=11, y=None,
              w=round(exp_size[1] / 6), h=round(exp_size[0] / 7))
    pdf.set_font('')
    pdf.set_font('Arial', size=10, style='B')
    pdf.ln(3)
    pdf.cell(80, 5, txt='Example Quality Control Visualisation', ln=1)
    pdf.ln(1)
    exp_size = io.imread(full_QC_model_path + '/Quality Control/QC_example_data.png').shape
    pdf.image(full_QC_model_path + '/Quality Control/QC_example_data.png', x=16, y=None, w=round(exp_size[1] / 8),
              h=round(exp_size[0] / 8))
    pdf.ln(1)
    pdf.set_font('')
    pdf.set_font('Arial', size=11, style='B')
    pdf.ln(1)
    pdf.cell(180, 5, txt='Quality Control Metrics', align='L', ln=1)
    pdf.set_font('')
    pdf.set_font_size(10.)

    pdf.ln(1)
    html = """
  <body>
  <font size="10" face="Courier New" >
  <table width=60% style="margin-left:0px;">"""
    with open(full_QC_model_path + '/Quality Control/QC_metrics_' + QC_model_name + '.csv', 'r') as csvfile:
        metrics = csv.reader(csvfile)
        header = next(metrics)
        image = header[0]
        IoU = header[1]
        IoU_OptThresh = header[2]
        header = """
    <tr>
    <th width = 33% align="center">{0}</th>
    <th width = 33% align="center">{1}</th>
    <th width = 33% align="center">{2}</th>
    </tr>""".format(image, IoU, IoU_OptThresh)
        html = html + header
        i = 0
        for row in metrics:
            i += 1
            image = row[0]
            IoU = row[1]
            IoU_OptThresh = row[2]
            cells = """
        <tr>
          <td width = 33% align="center">{0}</td>
          <td width = 33% align="center">{1}</td>
          <td width = 33% align="center">{2}</td>
        </tr>""".format(image, str(round(float(IoU), 3)), str(round(float(IoU_OptThresh), 3)))
            html = html + cells
        html = html + """</body></table>"""

    pdf.write_html(html)

    pdf.ln(1)
    pdf.set_font('')
    pdf.set_font_size(10.)
    ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "Democratising deep learning for microscopy with ZeroCostDL4Mic." Nature Communications (2021).'
    pdf.multi_cell(190, 5, txt=ref_1, align='L')
    ref_2 = '- Unet: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.'
    pdf.multi_cell(190, 5, txt=ref_2, align='L')

    pdf.ln(3)
    reminder = 'To find the parameters and other information about how this model was trained, go to the training_report.pdf of this model which should be in the folder of the same name.'

    pdf.set_font('Arial', size=11, style='B')
    pdf.multi_cell(190, 5, txt=reminder, align='C')

    pdf.output(full_QC_model_path + '/Quality Control/' + QC_model_name + '_QC_report.pdf')

    print('------------------------------')
    print('QC PDF report exported as ' + full_QC_model_path + '/Quality Control/' + QC_model_name + '_QC_report.pdf')


# Build requirements file for local run
after = [str(m) for m in sys.modules]
build_requirements_file(before, after)