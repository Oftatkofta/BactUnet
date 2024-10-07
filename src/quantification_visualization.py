# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:21:01 2022

@author: Jens
"""
import pandas as pd
import seaborn as sns
import os
import numpy as np

startpath = r"F:\BactUnet"
infile = os.path.join(startpath, 'bacteria_count_bactunet_V3.csv')
alldata = pd.read_csv(infile)
onlywt = alldata[alldata.bacteria == 'wt']


def hampel(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()    
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=np.nan
    return(vals)

g1 = sns.relplot(data=onlywt, x='frame', y='count', kind='line', hue='condition', col='method')
g1.set(ylim=(0,200))

g1 = sns.relplot(data=onlywt, x='frame', y='count', kind='line', hue='filename', row='method', col='condition')
g1.set(ylim=(0,200))

no_outliers = None

for m in onlywt.method.unique():
    sf = onlywt[onlywt['method']==m]
    for fn in onlywt.filename.unique():
        ssf = sf[(sf['filename']==fn)]
        cnts = ssf['count']
        ssf['count_hampel'] = hampel(cnts, 7, 5)
        if no_outliers is None:
            no_outliers = ssf
        no_outliers = pd.concat((no_outliers, ssf), ignore_index=True)
 
   
 
g2 = sns.relplot(data=no_outliers.dropna(), x='frame', y='count_hampel', kind='line', hue='condition', col='method')
g2.set(ylim=(0,200))