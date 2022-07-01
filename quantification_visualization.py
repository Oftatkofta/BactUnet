# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:21:01 2022

@author: Jens
"""
import pandas as pd
import seaborn as sns
import os

from matplotlib import pyplot as plt

startpath = r"F:\BactUnet"
infile = os.path.join(startpath, 'fl_quantifiaction_raw_data.csv')
alldata = pd.read_csv(infile)
onlywt = alldata[alldata.bacteria == 'wt']

infile = os.path.join(startpath,  "fl_quantifiaction_otsu_data.csv")
otsudata = pd.read_csv(infile)
otsuwt = otsudata[otsudata.bacteria == 'wt']

sns.set_theme(style="ticks")

g = sns.FacetGrid(col_wrap=4, col="filename", hue="condition", sharey=False, data=otsuwt)
g.map(sns.lineplot, "frame", "thresholded_median_intensity", alpha=.7)
#g.add_legend()

#sns.relplot(x="frame", y="norm_median_intensity", col="experiment", hue="condition", data=onlywt,
#           col_wrap=4, ci=None, palette="muted", height=4, sharex=False)



# =============================================================================
# sns.relplot(x="frame", y="raw_mean_intensity", hue="condition",
#             kind="line", 
#             data=onlywt)
# =============================================================================
plt.ylim(0, 0.4)
plt.show()