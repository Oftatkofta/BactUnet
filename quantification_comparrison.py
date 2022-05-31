# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:57:14 2022

@author: Jens
"""

from cellocity.channel import Channel, MedianChannel
from tifffile import TiffFile
import tifffile

myfile = r"C:\Users\Jens\Desktop\anodisc_ENR+RT_wt_1_MMStack_Default.ome.tif\anodisc_ENR+RT_wt_1_MMStack_Default.ome.tif"
print(tifffile.__version__)

with TiffFile(myfile) as tif:
    channel_1 = Channel(0, tif, "DIC") #0-indexed channels, meaning ch1 in ImageJ
    channel_2 = Channel(1, tif, "mCherry")
    pages = tif.pages
    print(tif.pages[0].tags)
    

##3-frame gliding temporal median projection by default
#channel_2_median = MedianChannel(channel_1)
#print(channel_1.finterval_ms, channel_2.pxSize_um)
#print(channel_2.pages)