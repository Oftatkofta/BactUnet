import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tifffile import TiffFile
import tifffile
import numpy as np
from preprocessing import normalizePercentile, unpatch_stack, predict_stack, patch_stack, patch_image, pad_stack, threshold_prediction_array
from window_functions import build_weighted_mask_array
from quantification_comparrison import list_files, get_metadata
from thresholded_object_counter import count_bacteria, bg_subtract, threshold_array

if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.')

else:
  print('You have GPU access')

# print the tensorflow version
print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

#load pretrained model
model = keras.models.load_model("models/bactunet_V4_3frame_1000ep.hdf5", compile=False)
model2 = keras.models.load_model("models/bactunet_V4_single_frame.hdf5", compile=False) #for frames 0 & 239
#set patch size
SIZE = 288
batch_size = 64
weight_mask_a = build_weighted_mask_array('hann', SIZE, 8)
weight_mask_b = 1 - weight_mask_a
threshold = 0.9
def _predict_arr_a(arr):
    """
    makes patch pattern A (standard) and predicts on 3frame and single frame models 
    """
    dic_arr_patch = patch_stack(arr, SIZE)
    first_frame_patch = patch_image(arr[0], SIZE)
    last_frame_patch = patch_image(arr[-1], SIZE)
    
    pred_arr = predict_stack(dic_arr_patch, batch_size, model)
    #print("prediction patch shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    pred_arr = unpatch_stack(pred_arr, 8, 8, 1)[:,0,:,:]
    #print("prediction array shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    
    first_frame_pred = unpatch_stack(predict_stack(first_frame_patch, batch_size, model2), 8, 8, 1)[:,0,:,:]
    last_frame_pred = unpatch_stack(predict_stack(last_frame_patch, batch_size, model2), 8, 8, 1)[:,0,:,:]
    #print("first frame prediction shape: {} type {} min/max {}/{}".format(first_frame_pred.shape, first_frame_pred.dtype, first_frame_pred.min(), first_frame_pred.max()))
    
    saveme = np.concatenate((first_frame_pred, pred_arr, last_frame_pred), axis=0)
    #print("full prediction shape: {} type {} min/max {}/{}".format(saveme.shape, saveme.dtype, saveme.min(), saveme.max()))
	
    return saveme
    
def _predict_arr_b(arr):
    """
    makes patch pattern B (padded) and predicts on 3frame ans single frame models, crops the outpud down to 2304x2304
    """
    arr = pad_stack(arr, SIZE)
    pad_SIZE = int(SIZE / 2)
    
    dic_arr_patch = patch_stack(arr, SIZE)
    first_frame_patch = patch_image(arr[0], SIZE)
    last_frame_patch = patch_image(arr[-1], SIZE)
    
    pred_arr = predict_stack(dic_arr_patch, batch_size, model)
    print("prediction patch shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    pred_arr = unpatch_stack(pred_arr, 9, 9, 1)[:,0,pad_SIZE:-pad_SIZE,pad_SIZE:-pad_SIZE]
    print("prediction array shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    
    first_frame_pred = unpatch_stack(predict_stack(first_frame_patch, batch_size, model2), 9, 9, 1)[:,0,pad_SIZE:-pad_SIZE,pad_SIZE:-pad_SIZE]
    last_frame_pred = unpatch_stack(predict_stack(last_frame_patch, batch_size, model2), 9, 9, 1)[:,0,pad_SIZE:-pad_SIZE,pad_SIZE:-pad_SIZE]
    print("first frame prediction shape: {} type {} min/max {}/{}".format(first_frame_pred.shape, first_frame_pred.dtype, first_frame_pred.min(), first_frame_pred.max()))
    
    saveme = np.concatenate((first_frame_pred, pred_arr, last_frame_pred), axis=0)
    print("full prediction shape: {} type {} min/max {}/{}".format(saveme.shape, saveme.dtype, saveme.min(), saveme.max()))
	
    return saveme
    


def process_one_file(metadata, stopframe=None):
    fh = metadata["filepath"]

    with TiffFile(fh) as tif:
        arr = tif.asarray()

    if stopframe is not None:
        arr = arr[0:stopframe, :, :, :]


    dic_arr = arr[:, 0, :, :]
    dic_arr = normalizePercentile(dic_arr, 0.1, 99.9, clip=True)
    
    pred_a = _predict_arr_a(dic_arr)
    pred_b = _predict_arr_b(dic_arr)
    
    #tifffile.imwrite(os.path.join(r"D:\Jens\BactUnet\32-bit_pred_AB", "AB_"+metadata["filename"]), np.stack((pred_a, pred_b), axis=1), imagej=True, resolution=(1. / 2.6755, 1. / 2.6755),metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})
	
    pred_merge = pred_a*weight_mask_a + pred_b*weight_mask_b
	
    pred_a = threshold_prediction_array(pred_a, threshold)
    pred_b = threshold_prediction_array(pred_b, threshold)
    pred_merge = threshold_prediction_array(pred_merge, threshold)
    dic_arr = dic_arr * 255
    dic_arr = dic_arr.astype('uint8')
    
    return np.stack((dic_arr, pred_a, pred_b, pred_merge), axis=1)


def run_analysis(infiles, stopframe=None):
    out_df = None
    for f in infiles:
        print("Working on ", f)
        fp = os.path.abspath(f)
        metadata = get_metadata(fp)
        arrs = process_one_file(metadata, stopframe)
        
       
        savepath = r"D:\Jens\BactUnet\optimal_prediction_AB_single"
        tifffile.imwrite(os.path.join(savepath, "AB_"+metadata["filename"]), arrs, imagej=True, resolution=(1. / 2.6755, 1. / 2.6755),
                 metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})

    return 1

if __name__ == '__main__':
    startpath = r"F:\BactUnet\bactunet_val"
    infiles = list_files(startpath, prettyPrint=False)
    run_analysis(infiles, stopframe=None)