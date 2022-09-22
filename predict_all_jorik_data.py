import pandas
import tensorflow as tf
from tensorflow import keras
import os
import tifffile as tiff
import numpy as np
from keras.models import load_model
from preprocessing import normalizePercentile, unpatch_stack, predict_stack
from quantification_comparrison import list_files
from thresholded_object_counter import count_bacteria, bg_subtract, threshold_array

if tf.test.gpu_device_name()=='':
    print('You do not have GPU access.')

else:
  print('You have GPU access')

# print the tensorflow version
print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

#load pretrained model
model = load_model("models/bactunet_noEmpty_final_alldata.hdf5", compile=False)
#set patch size
SIZE = 288






saveme = vikt * 65535
saveme = saveme.astype('uint16')
img = np.expand_dims(image[0:-2], axis = 1)
img = img * 65535
img = img.astype('uint16')
print(saveme.shape, img.shape)
saveme = np.concatenate((img,saveme), axis = 1)
print(saveme.shape)

def nbact_as_df(counts, method, metadata):
    """
    Returns frame, count, method and metadata for the file as a Pandas DataFrame.

    :return: DataFrame with 8 columns
    :rtype: pandas.DataFrame
    """

    tempdict = {
        "frame": range(1, len(raw_nbact) + 1),
        "count": counts,
        "method": method,
        "condition": metadata["condition"],
        "experiment": metadata["experiment"],
        "bacteria": metadata["bacteria"],
        "filepath": metadata["filepath"],
        "filename": metadata["filename"]
    }
    df = pd.DataFrame(tempdict)

    return df

def threshold_prediction_array(arr):
    out = arr > 0.1
    out = out * 255
    return out.astype('uint8')

def process_one_file(metadata, stopframe=None):
    fh = metadata["filepath"]

    with TiffFile(fh) as tif:
        arr = tif.asarray()

    if stopframe is not None:
        arr = arr[0:stopframe, :, :, :]




    dic_arr = arr[:, 0, :, :]
    dic_arr = normalizePercentile(dic_arr, 0.1, 99.9, clip=True)
    patched_dic_arr = patch_stack(dic_arr, SIZE)
    print("Original shape: {}, patched shape: {}".format(dic_arr.shape, patched_dic_arr.shape))
    pred_arr = predict_stack(patched_dic_arr, 2, model)
    pred_arr = unpatch_stack(pred_arr, 8, 8, 1)

    pred_arr = threshold_prediction_array(pred_arr) #now a 8-bit binary array (0,255)
    dic_arr = dic_arr * 255
    dic_arr = dic_arr.astype('uint8') #Both channels now 8-bit for saving

    pred_nbact = count_bacteria(pred)
    pred_df = nbact_as_df(pred_nbact, 'BactUnet_V3', metadata)

    mcherry_arr = arr[1:-1, 1, :, :]  # mCherry is always ch2
    mcherry_arr = bg_subtract(mcherry_arr)
    mcherry_arr = threshold_array(mcherry_arr)
    mcherry_nbact = count_bacteria(mcherry_arr)
    mcherry_df = nbact_as_df(mcherry_nbact, 'Thresholded_mcherry', metadata)

    df = pd.concat([pred_df, mcherry_df])

    saveme = np.concatenate((dic_arr, mcherry_arr, pred_arr), axis=1)



    tiff.imwrite(os.path.join(r"F:\BactUnet\prediction_output", "pred_"+metadata["filename"]), saveme, imagej=True, resolution=(1. / 2.6755, 1. / 2.6755),
                 metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})


    return df


def run_analysis(infiles, savepath, stopframe=None):
    out_df = None
    for f in infiles:
        print("Working on ", f)
        fp = os.path.abspath(f)
        metadata = get_metadata(fp)
        df = process_one_file(metadata, stopframe)
        if out_df is None:
            out_df = df

        out_df = pd.concat([out_df, df])

    out_df.to_csv(savepath, index=False)

    return out_df

if __name__ == '__main__':
    startpath = r"F:\BactUnet\bactunet_val"
    infiles = list_files(startpath, prettyPrint=False)
    savepath = r"F:\BactUnet\bacteria_count_bactunet_V3.csv"
    run_analysis(infiles, savepath, stopframe=14)