import os
import numpy as np
import pandas as pd
import seaborn as sns
from tifffile import TiffFile, imwrite
from scipy.ndimage import label


def minprojection(arr):
    out = np.zeros(arr.shape, dtype='uint8')  # 1 fewer frames than original

    for i in range(len(out)-2):
        out[i] = np.minimum(arr[i], arr[i + 1])
        out[i] = np.minimum(out[i], arr[i + 2])

    return out


def labelarr(arr):
    out = np.empty_like(arr)

    for i in range(len(arr)):
        labeled_arr, n_features = label(arr[i])
        out[i] = labeled_arr

    return out




workdir = os.path.abspath(r"F:\full predictions\dspi4")

infiles = [f for f in os.listdir(workdir) if ".tif" in f]

for f in infiles:
    fname = f.split("_")
    strain = fname[0]
    condition = fname[1]

    with TiffFile(os.path.join(workdir, f)) as tif:
        arr = tif.asarray()[:, 1, :, :]
        #arr = arr > 1
        minproj = minprojection(arr)
        proj_labs = labelarr(minproj)
        mask_labs = labelarr(arr)

        saveme = np.stack((mask_labs, proj_labs), axis=1)

        imwrite(os.path.join(r"F:\BactUnet\prediction_output", "labl_" + f), saveme, imagej=True,
                resolution=(1. / 2.6755, 1. / 2.6755),
                metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})