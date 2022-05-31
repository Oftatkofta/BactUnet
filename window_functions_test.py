# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:44:58 2022

@author: Jens Eriksson
"""

from window_functions import hann_window, corner_hann_window, top_hann_window, bartley_hann_window, triangular_window
from window_functions import build_weighted_mask_array
from matplotlib import pyplot as plt
import numpy as np

indir = r"C:\Users\Jens\Documents\Code\bactnet\Bactnet\Training data\stacks\predict\piccolo"

h = 288
w = 288


plt.subplot(331)
plt.imshow(corner_hann_window(h, w))
plt.subplot(332)
plt.imshow(top_hann_window(h, w))
plt.subplot(333)
plt.imshow(np.rot90(corner_hann_window(h, w), 3))
plt.subplot(334)
plt.imshow(np.rot90(top_hann_window(h, w), 1))
plt.subplot(335)
plt.imshow(hann_window(h, w))
plt.subplot(336)
plt.imshow(np.rot90(top_hann_window(h, w), 3))
plt.subplot(337)
plt.imshow(np.rot90(corner_hann_window(h, w), 1))
plt.subplot(338)
plt.imshow(np.rot90(top_hann_window(h, w), 2))
plt.subplot(339)
plt.imshow(np.rot90(corner_hann_window(h, w), 2))
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

print(build_weighted_mask_array("han", 3))
