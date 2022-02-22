import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import os

indir = "Bactnet\Training data\stacks\predict\piccolo"
outdir = "Bactnet\Training data\stacks\predict"
files = os.listdir(indir)

def _createOutArr(shape, nrows, ncols, nchannels):
    out_height = int(nrows * shape[-2])
    out_width = int(ncols * shape[-1])
    out_frames = int(shape[0] / (nrows * ncols))

    outshape = (out_frames, nchannels, out_height, out_width)

    out_arr = np.empty(outshape, dtype=np.float32)

    return out_arr

def unpatcher(arr, nrows, ncols, nchannels=1):
    out_arr = _createOutArr(arr.shape, nrows, ncols, nchannels)
    patch_h = arr.shape[-2]
    patch_w = arr.shape[-1]
    n = 0
    for frame in range(out_arr.shape[0]):
        for i in range(nrows):
            for j in range(ncols):
                y = patch_h * i
                x = patch_w * j
                out_arr[frame, :, y:y+patch_h, x:x+patch_w] = arr[n]
                n += 1

    return out_arr

for file in files:

    loadme = os.path.join(indir, file)
    saveme = os.path.join(outdir, file.split(".")[0] + "_unp.tif")
    with tiff.TiffFile(loadme) as tif:
        arr = tif.asarray()
        xres = tif.pages[0].tags['XResolution']
        print(type(xres))
    unpatched = unpatcher(arr, 8, 8, 2)
    print(saveme, unpatched.shape)
    tiff.imwrite(saveme, unpatched, imagej=True, resolution=(1./2.6755, 1./2.6755),
                 metadata={'unit': 'um', 'finterval': 15, 'axes': 'TCYX'})



