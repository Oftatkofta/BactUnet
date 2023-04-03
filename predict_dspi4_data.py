import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tifffile import TiffFile
import tifffile
import numpy as np
from keras.models import load_model
from preprocessing import normalizePercentile, unpatch_stack, predict_stack, patch_stack, patch_image
from quantification_comparrison import list_files, get_metadata


if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.')

else:
  print('You have GPU access')

# print the tensorflow version
print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

#load pretrained model
model = load_model("models/bactunet_V4_3frame_empty_250ep.hdf5", compile=False)
model_single = load_model("models/bactunet_V4_single_frame.hdf5", compile=False)
#set patch size
SIZE = 288


def threshold_prediction_array(arr):
    out = arr * 255
   
    return out.astype('uint8')

def process_one_file(metadata, savepath, stopframe=None):
    fh = metadata["filepath"]

    with TiffFile(fh) as tif:
        arr = tif.asarray()

    if stopframe is not None:
        arr = arr[0:stopframe, :, :, :]




    dic_arr = arr[:, 0, :, :]
    dic_arr = normalizePercentile(dic_arr, 0.1, 99.9, clip=True)
    patched_dic_arr = patch_stack(dic_arr, SIZE)
    
    pred_arr = predict_stack(patched_dic_arr, 128, model)
    
    first_last = np.concatenate((patch_image(dic_arr[0], SIZE), patch_image(dic_arr[-1], SIZE)), axis=0)
    pred_fl = predict_stack(first_last, 128, model_single)
    pred_fl = unpatch_stack(pred_fl, 8, 8, 1)
	
    print("prediction shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    pred_arr = unpatch_stack(pred_arr, 8, 8, 1)
    pred_arr = np.concatenate((pred_fl[0], pred_arr[:,0,:,:], pred_fl[1]), axis=0)
    print("prediction shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
    pred_arr = threshold_prediction_array(pred_arr) #now a 8-bit binary array (0,255)
    print("prediction shape: {} type {} min/max {}/{}".format(pred_arr.shape, pred_arr.dtype, pred_arr.min(), pred_arr.max()))
	
    dic_arr = dic_arr * 255
    dic_arr = dic_arr.astype('uint8') #Both channels now 8-bit for saving, remove first two frames


    saveme = np.stack((dic_arr, pred_arr), axis=1)

    print("Saving...shape of save array:{}".format(saveme.shape))

    tifffile.imwrite(os.path.join(savepath, "pred_"+metadata["filename"]), saveme, imagej=True, resolution=(1. / 0.107, 1. / 0.107),
                 metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})


    return 1


def run_analysis(infiles, savepath, stopframe=None):
    
    for f in infiles:
        print("Working on ", f)
        fp = os.path.abspath(f)
        metadata = get_metadata(fp)
        process_one_file(metadata, savepath, stopframe)
        

    return 1

if __name__ == '__main__':
    startpath = r"F:\BactUnet\dspi4data"
    infiles = list_files(startpath, prettyPrint=False)
    infiles2 = list_files(r"F:\BactUnet\bactunet_val", prettyPrint=False)
    infiles.extend([infiles2[0], infiles2[9], infiles2[10]]) #1 OGM 2 ENR wt
    savepath = r"F:\full predictions\dspi4"
    run_analysis(infiles, savepath, stopframe=None)