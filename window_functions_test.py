# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:44:58 2022

@author: Jens Eriksson
"""

from window_functions import hann_window, corner_hann_window, top_hann_window, bartley_hann_window, triangular_window
from window_functions import build_weighted_mask_array
from matplotlib import pyplot as plt
import numpy as np

#indir = r"C:\Users\Jens\Documents\Code\bactnet\Bactnet\Training data\stacks\predict\piccolo"

h = 288
w = 288


img = build_weighted_mask_array("hann", 288, 3)
#img2 = 1-img
#img=img+img2

plt.imshow(img)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()

