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

def mean_intensities_as_df(raw_mean, raw_median, norm_mean, norm_median, metadata):
    """
    Returns frame, mean intensise and metadata for the file as a Pandas DataFrame.

    :return: DataFrame with 7 columns 
    :rtype: pandas.DataFrame
    """
    
    tempdict = {
                "frame" : range(1, raw_mean.shape[0]+1),
                "raw_mean_intensity":raw_mean,
                "temporal_median_intensity":raw_median,
                "norm_mean_intensity":norm_mean,
                "norm_median_intensity":norm_median,
                "condition":metadata["condition"],
                "experiment":metadata["experiment"], 
                "bacteria":metadata["bacteria"],
                "filename":metadata["filename"]
                }
    df = pd.DataFrame(tempdict)
    
    return df
    

def list_files(startpath, prettyPrint=True):
    """
    Lists and optionally prints the files in starpath.
    Return a list of the full paths to first .ome.tif files it enounters
    """
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
            if "_1.ome.tif" in f:
                break
            if ".tif" in f:
                out.append(root+"\\" + f)
            

    return out

def get_metadata(filepath):
    #returns a dictionary of metadata from the folder structure of the filepath
    out = {}
    metalist = filepath.split("\\")
    if len(metalist) > 7:
        out["condition"] = metalist[2]
        out["experiment"] = metalist[3]
        out["bacteria"] = metalist[4]
        out["filename"] = metalist[6]
        out["filepath"] = filepath
    
    return out
    

def process_one_file(metadata, stopframe=None):
    fh = metadata["filepath"]
    with TiffFile(fh) as tif:
        arr = tif.asarray()
        
    if stopframe is not None:
        arr = arr[0:stopframe,:,:,:]
    
    mcherry_arr = arr[:,1,:,:] #mCherry is always ch2
    norm_mcherry_arr = normalizeMinMax(mcherry_arr)
    
    mcherry_median = getTemporalMedianFilter(mcherry_arr,
                                             doGlidingProjection=True,
                                             startFrame=0,
                                             stopFrame=mcherry_arr.shape[0])
    
    norm_mcherry_median = normalizeMinMax(mcherry_median)
    
    mcherry_arr = mcherry_arr[1:-1, :, :] #drop first and frame to equate length
    norm_mcherry_arr = norm_mcherry_arr[1:-1, :, :] #drop first and frame to equate length
    
    raw_means = calculateMeanIntensity(mcherry_arr)
    raw_median = calculateMeanIntensity(mcherry_median)
    norm_means = calculateMeanIntensity(norm_mcherry_arr)
    norm_median = calculateMeanIntensity(norm_mcherry_median)
    
    print(raw_means.shape, raw_median.shape, norm_means.shape, norm_median.shape)
    out_df = mean_intensities_as_df(raw_means, raw_median, norm_means, norm_median ,metadata)
        
    return out_df
        
def run_analysis(infiles, savepath):
    out_df = None
    for f in infiles:
        print("Working on ", f)
        fp = os.path.abspath(f)
        metadata = get_metadata(fp)
        df = process_one_file(metadata, 12)
        if out_df is None:
            out_df = df
        
        out_df = pd.concat([out_df,df])
    
    out_df.to_csv(savepath, index=False)
    
    return out_df
    


startpath = r"F:\BactUnet\bactunet_val"
infiles = list_files(startpath, prettyPrint=False)
outfile = os.path.join(r"F:\BactUnet", 'fl_quantifiaction_raw_data.csv')
run_analysis(infiles, outfile)



     