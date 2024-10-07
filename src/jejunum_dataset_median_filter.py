# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:05:24 2022

@author: analyst
"""
import os
from tifffile import TiffFile
import tifffile
import numpy as np
#from quantification_comparrison import list_files, get_metadata

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

def get_metadata(filepath, lvl=4):
    #returns a dictionary of metadata from the folder structure of the filepath
    out = {}
    metalist = filepath.split("\\")
    #print(metalist)
    
    out["condition"] = metalist[lvl]
    out["experiment"] = metalist[lvl+1]
    out["bacteria"] = metalist[lvl+2]
    out["filename"] = metalist[lvl+4]
    out["filepath"] = filepath
    
    return out


def normalization_to_8bit(image_stack, lowPcClip=0.175, highPcClip=0.175):
    """
    Function to rescale 16/32/64 bit arrays to 8-bit for visualizing output
    Defaults to saturate 0.35% of pixels, 0.175% in each end by default, which often produces nice results. This
    is the same as pressing 'Auto' in the ImageJ contrast manager. `numpy.interp()` linear interpolation is used
    for the mapping.
    
    :param image_stack: 3D Numpy array to be rescaled
    :type image_stack: Numpy array
    :param lowPcClip: Fraction for black clipping bound
    :type lowPcClip: float
    :param highPcClip: Fraction for white/saturated clipping bound
    :type highPcClip: float
    :return: 8-bit numpy array of the same shape as image_stack
    :rtype: numpy.dtype('uint8')
    
    """

    # clip image to saturate 0.35% of pixels 0.175% in each end by default.
    low = int(np.percentile(image_stack, lowPcClip))
    high = int(np.percentile(image_stack, 100 - highPcClip))
    
    # use linear interpolation to find new pixel values
    image_equalized = np.interp(image_stack.flatten(), (low, high), (0, 255))
    
    return image_equalized.reshape(image_stack.shape).astype('uint8')

def process_one_file(metadata, stopframe=None):
    fh = metadata["filepath"]
    with TiffFile(fh) as tif:
        arr = tif.asarray()
        
    if stopframe is not None:
        arr = arr[0:stopframe,:,:,:]
    
    dic_arr = arr[:,0,:,:]
    mcherry_arr = arr[:,1,:,:] #mCherry is always ch2
    
    
    dic_arr = getTemporalMedianFilter(dic_arr,
                                             doGlidingProjection=True,
                                             startFrame=0,
                                             stopFrame=dic_arr.shape[0])
    
    
    mcherry_arr = getTemporalMedianFilter(mcherry_arr,
                                             doGlidingProjection=True,
                                             startFrame=0,
                                             stopFrame=mcherry_arr.shape[0])
    
    dic_arr = normalization_to_8bit(dic_arr)
    mcherry_arr = normalization_to_8bit(mcherry_arr)
    
    #print(dic_arr.shape, mcherry_arr.shape)
    
    dic_arr = np.expand_dims(dic_arr, axis=1)
    mcherry_arr = np.expand_dims(mcherry_arr, axis=1)
    
    out_arr = np.concatenate((dic_arr, mcherry_arr), axis=1)
        
    return out_arr



startpath = r"C:\Users\analyst\Desktop\Fileset_74326"
infiles = list_files(startpath, prettyPrint=False)
for f in infiles:
    print("working on: ", f)
    metadata = get_metadata(f)
    outfile = os.path.join(startpath, metadata['filename'])    
    
    out = process_one_file(metadata, stopframe=None)
    print("saving :", outfile)
    tifffile.imwrite(outfile, out, imagej=True, resolution=(1./0.109, 1./0.109),
                     metadata={'spacing': 1.0, 'unit': 'um', 'finterval': 15,
                               'axes': 'TCYX'})

#run_analysis(infiles, outfile)
