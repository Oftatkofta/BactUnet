import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tifffile import TiffFile
import tifffile
import numpy as np
from keras.models import load_model
from preprocessing import normalizePercentile, unpatch_stack, predict_stack, patch_stack
from quantification_comparrison import list_files
from thresholded_object_counter import count_bacteria, bg_subtract, threshold_array, get_metadata

if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.')

else:
  print('You have GPU access')

# print the tensorflow version
print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

#load pretrained model
model = load_model(r"F:\BactUnet\models\bactunet_V4_single_frame.hdf5", compile=False)
#set patch size
SIZE = 288


def threshold_prediction_array(arr):
    out = arr > 0.2
    out = out * 255
    return out.astype('uint8')

def process_one_file(fname, stopframe=None):
    fh = fname

    with TiffFile(fh) as tif:
        arr = tif.asarray()

    if stopframe is not None:
        arr = arr[0:stopframe, :, :, :]

    dic_arr = arr[:, 0, :, :]
    dic_arr = normalizePercentile(dic_arr, 0.1, 99.9, clip=True)
    patched_dic_arr = patch_stack(dic_arr, SIZE)
    
    pred_arr = predict_stack(patched_dic_arr, 16, model)
    print("prediction shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    pred_arr = unpatch_stack(pred_arr, 8, 8, 1)
    print("prediction shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    pred_arr = threshold_prediction_array(pred_arr)[:,0,:,:] #now a 8-bit binary array (0,255)
    print("prediction shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
	
    #dic_arr = dic_arr * 255
    #dic_arr = dic_arr.astype('uint8')[1:-1,:,:] #Both channels now 8-bit for saving, remove first two frames
    #print("Original shape: {} {}, prediction shape: {} type {} min/max {}/{}".format(dic_arr.shape, dic_arr.dtype, pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    
    saveme = np.stack((pred_arr), axis=1)

    print("Saving...shape of save array:{}".format(saveme.shape))

    tifffile.imwrite(os.path.join(r"F:\BactUnet\prediction_output", "pred_V4_single"+fname), saveme, imagej=True, resolution=(1. / 0.167, 1. / 0.167),
                 metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})


    return True



if __name__ == '__main__':
    
    infile = r"F:\BactUnet\bactunet_val\OGM\EXP-20-BT0353\wt\OGM3_1_MMStack_Pos0.ome.tif\OGM3_1_MMStack.ome.tif"
    process_one_file(infile)