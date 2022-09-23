import numpy as np
import pandas as pd
import tifffile
from tifffile import TiffFile
from skimage.restoration import rolling_ball, ellipsoid_kernel
from skimage.filters import gaussian
from quantification_comparrison import getTemporalMedianFilter, get_metadata, list_files
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation
from skimage.measure import find_contours
import os



def get_both_arrays(fh):

    with TiffFile(fh) as tif:
        arr = tif.asarray()

    med_arr = getTemporalMedianFilter(arr[:,1,:,:],doGlidingProjection=True, startFrame=0, stopFrame=arr.shape[0])
    arr = arr[1:-1,1,:,:]

    return arr, med_arr

kernel = ellipsoid_kernel((25, 25), 75)

def apply_rolling_ball(frame):
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
    return bg_arr

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


def nbact_as_df(raw_nbact, med_nbact, metadata):
    """
    Returns frame, mean intensities and metadata for the file as a Pandas DataFrame.

    :return: DataFrame with 8 columns
    :rtype: pandas.DataFrame
    """

    tempdict = {
        "frame": range(1, len(raw_nbact) + 1),
        "raw_nbact": raw_nbact,
        "TM_nbact": med_nbact,
        "condition": metadata["condition"],
        "experiment": metadata["experiment"],
        "bacteria": metadata["bacteria"],
        "filepath": metadata["filepath"],
        "filename": metadata["filename"]
    }
    df = pd.DataFrame(tempdict)

    return df

def process_one_file(metadata):
    fh = metadata["filepath"]

    arrs = get_both_arrays(fh)
    bg_arr = bg_subtract(arrs[0])
    bg_med = bg_subtract(arrs[1])

    bg_arr = threshold_array(bg_arr)
    bg_med = threshold_array(bg_med)

    raw_nbact = count_bacteria(bg_arr)
    med_nbact = count_bacteria(bg_med)

    tifffile.imwrite(os.path.join(r"F:\BactUnet\thresholded_mcherry", "raw_"+metadata["filename"]), bg_arr)
    tifffile.imwrite(os.path.join(r"F:\BactUnet\thresholded_mcherry", "TM_"+metadata["filename"]), bg_med)

    df = nbact_as_df(raw_nbact, med_nbact, metadata)

    return df


def run_analysis(infiles, savepath):
    out_df = None
    for f in infiles:
        print("Working on ", f)
        fp = os.path.abspath(f)
        metadata = get_metadata(fp)
        df = process_one_file(metadata)
        if out_df is None:
            out_df = df

        out_df = pd.concat([out_df, df])

    out_df.to_csv(savepath, index=False)

    return out_df

if __name__ == '__main__':
    startpath = r"F:\BactUnet\bactunet_val"
    infiles = list_files(startpath, prettyPrint=False)
    savepath = r"F:\BactUnet\bacteria_count_V2.csv"
    run_analysis(infiles, savepath)