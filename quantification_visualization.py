# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:21:01 2022

@author: Jens
"""
import pandas as pd
import seaborn as sns
import os

from matplotlib import pyplot as plt

startpath = r"F:\bactunet_val"
infile = os.path.join(startpath, 'fl_quantifiaction_raw_data.csv')
alldata = pd.read_csv(infile)

sns.relplot(x="frame", y="temporal_median_intensity", hue="filename",style="bacteria", kind="line", data=alldata)

