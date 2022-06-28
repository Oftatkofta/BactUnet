# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:57:14 2022

@author: Jens
"""

from cellocity.channel import Channel, MedianChannel
from tifffile import TiffFile
import tifffile
import numpy as np
import pandas as pd
from preprocessing import normalizeMinMax
import seaborn as sns
import os

from matplotlib import pyplot as plt

print(tifffile.__version__)



def getTemporalMedianFilter(arr, doGlidingProjection, startFrame, stopFrame,
                            frameSamplingInterval=3):
    """
    Returns a temporal median filter of the array.

    The function runs a gliding N-frame temporal median on every pixel to smooth out noise and to remove fast moving
    debris that is not migrating cells.

    :param doGlidingProjection: Should a gliding (default) or binned projection be performed?
    :type doGlidingProjection: bool
    :param stopFrame: Last frame to analyze, defaults to analyzing all frames if ``None``.
    :type stopFrame: int
    :param startFrame: First frame to analyze.
    :type startFrame: int
    :param frameSamplingInterval: Do median projection every N frames.
    :type frameSamplingInterval: int
    :return: Numpy array
    :rtype: numpt.ndarray

    """

    if (startFrame >= stopFrame):
        raise ValueError("StartFrame cannot be larger than or equal to Stopframe!")

    if (stopFrame - startFrame < frameSamplingInterval):
        raise ValueError("Not enough frames selected to do median projection! ")

    if doGlidingProjection:
        nr_outframes = (stopFrame - startFrame) - (frameSamplingInterval - 1)

    else:
        nr_outframes = int((stopFrame - startFrame) / frameSamplingInterval)

    outshape = (nr_outframes, arr.shape[1], arr.shape[2])
    outframe = 0

    # Filling a pre-created array is computationally cheaper
    out = np.ndarray(outshape, dtype=np.float32)

    if doGlidingProjection:
        for inframe in range(startFrame, stopFrame - frameSamplingInterval + 1):
            # median of frames n1,n2,n3...
            frame_to_store = np.median(arr[inframe:inframe + frameSamplingInterval], axis=0).astype(np.float32)

            out[outframe] = frame_to_store
            outframe += 1
    else:
        for inframe in range(startFrame, stopFrame, frameSamplingInterval):
            # median of frames n1,n2,n3...
            frame_to_store = np.median(arr[inframe:inframe + frameSamplingInterval], axis=0).astype(np.float32)

            if outframe > nr_outframes - 1:
                break
            out[outframe] = frame_to_store
            outframe += 1

    return out

def calculateMeanIntensity(arr):
    """
    Calculates the average intensity for each time point in array

    :return: mean values for array
    :rtype: 1D numpy.ndarray of the same length as self.speeds

    """

    if np.isnan(arr).any():
        avg_intensity = np.nanmean(arr, axis=(1, 2))

    else:
        avg_intensity = arr.mean(axis=(1, 2))

    return avg_intensity

def meanIntensityAsDf(arr, colname="MEAN_intensity_AU"):
    """
    Returns frame and mean intensity for the frame as a Pandas DataFrame.

    :return: DataFrame with 1 column for average speed and index = frame number
    :rtype: pandas.DataFrame
    """
    avg_intensities = calculateMeanIntensity(arr)
    df = pd.DataFrame(avg_intensities, columns=[colname])
    df['frame'] = range(0, len(df))
    return df



def spaghetti(myfile):
    with TiffFile(myfile) as tif:
        arr = tif.asarray()
        arr = arr[0:12,:,:,:]
        mcherry_arr = arr[:,1,:,:]
        mcherry_arr = normalizeMinMax(mcherry_arr)
        mcherry_median = getTemporalMedianFilter(mcherry_arr, doGlidingProjection=True, startFrame=0, stopFrame=mcherry_arr.shape[0])
        mcherry_arr = mcherry_arr[1:, :, :] #drop firs frame to equate length
        print(arr.shape, mcherry_arr.shape, mcherry_median.shape)
        mean_df = meanIntensityAsDf(mcherry_arr, "raw_mean")
        mean_df = mean_df.melt(id_vars=(["frame"]), value_vars=(["raw_mean"]))
        filter_mean_df = meanIntensityAsDf(mcherry_median, "temporal_median_mean").melt(id_vars=(["frame"]), value_vars=(["temporal_median_mean"]))
        df = pd.concat([mean_df, filter_mean_df])
        print(df)
        sns.scatterplot(x="frame", y="value",hue="variable", data=df)

    

def list_files(startpath, prettyPrint=True):
    out = []
    for root, dirs, files in os.walk(startpath):
        if prettyPrint:
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)

        for f in files:
            if prettyPrint:
                print('{}{}'.format(subindent, f))
                if ".ome.tif" in f:
                    print(root+"\\" + f)
            out.append(root+"\\" + f)
            break

    return out
startpath = r"F:\bactunet_val"
infiles = list_files(startpath, prettyPrint=True)
# for f in infiles:
#     fp = os.path(f)
#     print(fp)