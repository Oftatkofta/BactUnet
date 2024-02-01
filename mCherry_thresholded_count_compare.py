import numpy as np
import pandas as pd
import tifffile
from tifffile import TiffFile
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation
from skimage.measure import find_contours
from skimage.restoration import rolling_ball, ellipsoid_kernel
from skimage.filters import gaussian

import os

def apply_rolling_ball(frame, kernel=None):
    if kernel is None:
        kernel = ellipsoid_kernel((25, 25), 75)
    background = rolling_ball(frame, kernel=kernel)
    filtered_image = frame - background
    return filtered_image

def bg_subtract(arr):
    bg_arr = []
    for i in range(len(arr)):
        print("rolling with frame: ", i)
        a = gaussian(arr[i], sigma=0.8, preserve_range=True)
        a = apply_rolling_ball(a)
        bg_arr.append(a)
    bg_arr = (bg_arr/np.max(bg_arr))*255

    return bg_arr.astype('uint8')

def count_bacteria(arr):

    bact_per_frame = []

    for frame in arr:
        nbact = 0
        contours = find_contours(frame)
        for c in contours:
            if c.size > 100:
                nbact += 1
        bact_per_frame.append(nbact)

    return bact_per_frame

def threshold_array(arr):
    out = np.empty_like(arr)

    for i in range(len(arr)):
        frame = arr[i]
        th = threshold_otsu(frame)
        frame = frame > th*1.5
        frame = binary_erosion(frame)
        frame = binary_erosion(frame)
        frame = binary_dilation(frame)

        out[i] = frame
    out = out * 255

    return out.astype('uint8')


infiles = {
        "BT0398_Ch2":r"C:\Users\Jens\Documents\Code\BactUnet\Bactnet\BT0398_OGM\C2-anodisc_OGM_1_MMStack_Default.ome.tif",
        "BT402_Ch2":r"C:\Users\Jens\Documents\Code\BactUnet\Bactnet\BT402_ENR_wt\C2-BT0402-anodisc_ENR_wt_1_MMStack_Default.tif",
        "BT403_Ch2":r"C:\Users\Jens\Documents\Code\BactUnet\Bactnet\BT403_OGM_wt\C2-anodisc_OGM_wt_1_MMStack_Default.ome.tif",
        "BT404_Ch2":r"C:\Users\Jens\Documents\Code\BactUnet\Bactnet\BT404_ENRRT_wt\C2_anodisc_ENR+RT_wt_1_MMStack_Default_1.ome.tif"
            }

outdir = r"C:\Users\Jens\Documents\Code\BactUnet\Bactnet\mcherry_count_temp"

kernel = ellipsoid_kernel((25, 25), 75)
counts = None

for k in infiles.keys():
    with TiffFile(infiles[k]) as tif:
        arr =  tif.asarray()[1:-1,:,:]
    bg_arr = bg_subtract(arr)
    bg_arr = threshold_array(bg_arr)
    raw_nbact = count_bacteria(bg_arr)
    tifffile.imwrite(os.path.join(outdir, k+".tif"), bg_arr)
    df = pd.DataFrame({k:raw_nbact})
    if counts is None:
        counts = df
    else:
        counts = pd.concat((counts,df), axis=1)
counts.to_csv(os.path.join(outdir,"counts.csv"), index=False)