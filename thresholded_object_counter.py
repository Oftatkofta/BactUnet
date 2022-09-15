import numpy as np
from tifffile import TiffFile
from skimage.restoration import rolling_ball, ellipsoid_kernel
from skimage.filters import gaussian, threshold_otsu
from quantification_comparrison import getTemporalMedianFilter, list_files, get_metadata

startpath = r"F:\BactUnet\bactunet_val"
infiles = list_files(startpath, prettyPrint=False)




with TiffFile(fh) as tif:
    arr = tif.asarray()



image = arr[50]
image_gray = arr[50]

#blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

#blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
#blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()