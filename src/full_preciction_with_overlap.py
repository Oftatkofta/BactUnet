import argparse
import tensorflow as tf
from tensorflow import keras
import os
from tifffile import TiffFile
import tifffile
import numpy as np
from preprocessing import normalizePercentile, unpatch_stack, predict_stack, patch_stack, patch_image, pad_stack, threshold_prediction_array
from window_functions import build_weighted_mask_array
from quantification_comparrison import get_metadata
from mCherry_thresholded_count_compare import bg_subtract


def test_gpu_access():
    # Check if GPU is available for use
    if not tf.config.list_physical_devices('GPU'):
        print('You do not have GPU access.')
        return False
    else:
        print('You have GPU access')
        return True

# Print the TensorFlow and Keras version
print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

# Load pretrained models, one using 3 frames and one using single frames
model_3frame = keras.models.load_model("models/bactunet_V4_3frame.hdf5", compile=False)
model_single_frame = keras.models.load_model("models/bactunet_V4_single_frame.hdf5", compile=False)  # for frames 0 & 239

# Set patch size
SIZE = 288
batch_size = 64

# Create mask for combining patch A & B predictions
weight_mask_a = build_weighted_mask_array('hann', SIZE, 8)
weight_mask_b = 1 - weight_mask_a
threshold = 0.9

def predict_array(arr, is_padded=False):
    """
    Predict on given array using both single frame and 3-frame models.
    Optionally pads the input array and crops the output accordingly.
    """
    if is_padded:
        arr = pad_stack(arr, SIZE)
        pad_SIZE = int(SIZE / 2)
    
    dic_arr_patch = patch_stack(arr, SIZE)
    first_frame_patch = patch_image(arr[0], SIZE)
    last_frame_patch = patch_image(arr[-1], SIZE)

    pred_arr = predict_stack(dic_arr_patch, batch_size, model_3frame)
    pred_arr = unpatch_stack(pred_arr, 9 if is_padded else 8, 9 if is_padded else 8, 1)[:, 0, pad_SIZE:-pad_SIZE, pad_SIZE:-pad_SIZE] if is_padded else unpatch_stack(pred_arr, 8, 8, 1)[:, 0, :, :]

    first_frame_pred = unpatch_stack(predict_stack(first_frame_patch, batch_size, model_single_frame), 9 if is_padded else 8, 9 if is_padded else 8, 1)[:, 0, pad_SIZE:-pad_SIZE, pad_SIZE:-pad_SIZE] if is_padded else unpatch_stack(predict_stack(first_frame_patch, batch_size, model_single_frame), 8, 8, 1)[:, 0, :, :]
    last_frame_pred = unpatch_stack(predict_stack(last_frame_patch, batch_size, model_single_frame), 9 if is_padded else 8, 9 if is_padded else 8, 1)[:, 0, pad_SIZE:-pad_SIZE, pad_SIZE:-pad_SIZE] if is_padded else unpatch_stack(predict_stack(last_frame_patch, batch_size, model_single_frame), 8, 8, 1)[:, 0, :, :]

    saveme = np.concatenate((first_frame_pred, pred_arr, last_frame_pred), axis=0)
    return saveme

def process_one_file(image_path, output_path, stopframe=None):
    try:
        with TiffFile(image_path) as tif:
            arr = tif.asarray()
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        return
    except Exception as e:
        print(f"Error: Could not open file {image_path}. Exception: {e}")
        return

    # Slice array if stopframe is provided
    if stopframe is not None:
        arr = arr[0:int(stopframe), :, :, :]

    dic_arr = normalizePercentile(arr[:, 0, :, :], 0.1, 99.9, clip=True)
    mcherry_arr = bg_subtract(arr[:, 1, :, :])

    # Predict using both patch patterns A and B
    pred_a = predict_array(dic_arr, is_padded=False)
    pred_b = predict_array(dic_arr, is_padded=True)

    # Merge predictions with weight masks
    pred_merge = pred_a * weight_mask_a + pred_b * weight_mask_b
    pred_merge = threshold_prediction_array(pred_merge, threshold)
    dic_arr = (dic_arr * 255).astype('uint8')
    arrs = np.stack((dic_arr, pred_merge, mcherry_arr), axis=1)

    # Intensity value range
    val_range = np.arange(256, dtype=np.uint8)
    # Gray LUT
    lut_gray = np.stack([val_range, val_range, val_range])
    # Red LUT
    lut_red = np.zeros((3, 256), dtype=np.uint8)
    lut_red[0, :] = val_range
    # Green LUT
    lut_green = np.zeros((3, 256), dtype=np.uint8)
    lut_green[1, :] = val_range
    # Create ijmetadata kwarg
    ijmeta = {'LUTs': [lut_gray, lut_green, lut_red]}

    # Save the processed TIFF image
    tifffile.imwrite(output_path, arrs, imagej=True,
                     resolution=(1. / 2.6755, 1. / 2.6755),
                     metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX',
                               'mode': 'composite'},
                     ijmetadata=ijmeta)

def run_analysis(infiles, stopframe=None):
    for f in infiles:
        print("Working on", f)
        fp = os.path.abspath(f)
        output_path = os.path.join(r"D:\Jens\BactUnet\optimal_prediction_AB_single", os.path.basename(f))
        try:
            process_one_file(fp, output_path, stopframe)
        except Exception as e:
            print(f"Error processing file {f}: {e}")

def main():
    parser = argparse.ArgumentParser(description='BactUnet Image Processor')
    parser.add_argument('image_path', help='Path to the input TIFF image')
    parser.add_argument('output_path', help='Path to save the processed TIFF image')
    parser.add_argument('--stopframe', type=int, help='Frame to stop at')
    args = parser.parse_args()

    # Convert args to a dictionary to pass to the processing function
    process_one_file(**vars(args))

if __name__ == "__main__":
    main()