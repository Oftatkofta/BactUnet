import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

val_dir = r"C:\Users\Jens\Documents\Code\BactUnet\_results\network_compare\validation"
train_dir = r"C:\Users\Jens\Documents\Code\BactUnet\_results\network_compare\training"

def read_data(path, label):
    out = pd.DataFrame()
    #out_count = pd.DataFrame()
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith('TP.csv'):
                name = entry.name.split("_")
                fname = name[0]+"_"+name[1]
                network = name[2]
                
                tp_df = pd.read_csv(entry.path)
                gt_df = pd.read_csv(path+"\\"+fname+"_GT_count.csv")
                count_df = pd.read_csv(path+"\\"+fname+"_"+network+"_count.csv")
                frames = np.array(count_df.index) + int(name[1])
                count_df['frame'] =  frames
                count_df['file'] = fname
                count_df['network'] = network
                count_df['dataset'] = label
                count_df["TP"] = tp_df["Count"]
                count_df["GT"] = gt_df["Count"]
                count_df["FP"] = count_df["Count"] - count_df["TP"]
                count_df["FN"] = count_df["GT"] - count_df["TP"]
                count_df['precision'] = count_df['TP']/(count_df['TP']+count_df['FP'])
                count_df['recall'] = count_df['TP']/(count_df['TP']+count_df['FN'])
                count_df['average_precision'] = count_df['TP']/(count_df['TP']+count_df['FP']+count_df['FN'])
                
                #if counttype == 'TP':
                #    out_TP = pd.concat([out_TP, df], ignore_index=True)
                #else:
                #    out_count = pd.concat([out_count, df], ignore_index=True)
                out = pd.concat([out, count_df], ignore_index=True)
        return out 

alldata = pd.concat([read_data(val_dir, "validation"), read_data(train_dir, "train")], ignore_index=True)
print(alldata.head(), alldata.shape)

df = alldata.filter(items=['Count', 'Average Size','frame','file', 'network', 'dataset', 'TP', 'GT', 'FP','FN', 'precision', 'recall', 'average_precision'])
df.head()

print(df.network.unique())
print(df.dataset.unique())
print(df.file.unique())

df.groupby(['file', 'dataset']).sum()

plt.figure(figsize=(20,10))
#g1 = sns.boxplot(data=df, x='file', y='average_precision', hue='network')
#g2 = sns.boxplot(data=df, x='file', y='TP', hue='network')
g3 = sns.boxplot(data=df, x='file', y='recall', hue='network', palette='Set2').set(title='Recall')

plt.figure(figsize=(8,10))
g3 = sns.violinplot(data=df, x='dataset', y='average_precision', hue='network', palette="Set2").set(title='Average Precision')

plt.figure(figsize=(8,10))
g3 = sns.violinplot(data=df, x='dataset',hue='network', y='precision', palette="Set2").set(title='Precision')

plt.figure(figsize=(8,10))
g3 = sns.violinplot(data=df, x='dataset',hue='network', y='recall', palette="Set2").set(title='Recall')

df['F1-score'] = 2*df['precision']*df['recall']/(df['precision']+df['recall'])

plt.figure(figsize=(8,10))
g3 = sns.violinplot(data=df, x='dataset',hue='network', y='F1-score', palette="Set2").set(title='F1-score')

count_compare = {}
threshold = 0.5 

def countmask(mask):
    #counts contours in binary 3D-array
    out=[]
    for frame in range(len(mask)):
        out.append(len(measure.find_contours(mask[frame])))
    return out
    
    
for stack in image_dict.keys():
    count_compare[stack] = {}
    mask_true = image_dict[stack]['y_true']>1
    mask_single = image_dict[stack]["y_pred_single"][:,0,:,:]>threshold
    mask_3frame = image_dict[stack]["y_pred_3frame"][:,0,:,:]>threshold
    mask_empty = image_dict[stack]["y_pred_empty"][:,0,:,:]>threshold
    
    and_single = np.logical_and(mask_true, mask_single)
    and_3frame = np.logical_and(mask_true, mask_3frame)
    and_empty = np.logical_and(mask_true, mask_empty)
    
    single_and_3frame = np.logical_and(mask_single, mask_single)
     
    count_compare[stack]['frame'] = list(range(len(mask_true)))
    count_compare[stack]['true'] = countmask(mask_true)
    count_compare[stack]['single'] = countmask(mask_single)
    count_compare[stack]['3frame'] = countmask(mask_3frame)
    count_compare[stack]['empty'] = countmask(mask_empty)
    count_compare[stack]['true_and_single'] = countmask(and_single)
    count_compare[stack]['true_and_3frame'] = countmask(and_3frame)
    count_compare[stack]['true_and_empty'] = countmask(and_empty)
    count_compare[stack]['3frame_and_single'] = countmask(single_and_3frame)
    
count_df = None
#print(count_compare)
for stack in count_compare.keys():
    df = pd.DataFrame.from_dict(count_compare[stack])
    df['file'] = stack
    
    
    if count_df is None:
        count_df = df
    
    else:
        count_df = pd.concat([count_df, df], ignore_index=True)
count_df.head()

g1 = sns.boxplot(data=count_df, x='file', y='true_and_single', color='blue')
g1 = sns.boxplot(data=count_df, x='file', y='true_and_3frame', color='yellow')
g1 = sns.boxplot(data=count_df, x='file', y='true_and_empty', color='magenta')

count_df['TP_fraction_single'] = count_df['true_and_single']/count_df['true']
count_df['TP_fraction_3frame'] = count_df['true_and_3frame']/count_df['true']
count_df['TP_fraction_empty'] = count_df['true_and_empty']/count_df['true']

count_df['FP_fraction_single'] = (count_df['single']-count_df['true_and_single'])/count_df['single']
count_df['FP_fraction_3frame'] = (count_df['3frame']-count_df['true_and_3frame'])/count_df['3frame']
count_df['FP_fraction_empty'] = (count_df['empty']-count_df['true_and_empty'])/count_df['empty']

plt.figure(figsize=(20,10))
g1 = sns.boxplot(data=count_df, x='file', y='TP_fraction_single', color='white')
g1 = sns.boxplot(data=count_df, x='file', y='TP_fraction_3frame', color='blue')
g1 = sns.boxplot(data=count_df, x='file', y='TP_fraction_empty', color='green')


#Let's reformat the dataframe for easier plottting

count_df.head()

single = count_df.filter(items=['file', 'frame'])
single['model'] = 'single_frame'
single['TP'] = count_df['true_and_single']
single['FN'] = count_df['true']-count_df['true_and_single']
single['FP'] = count_df['single']-count_df['true_and_single']

multi = count_df.filter(items=['file', 'frame'])
multi['model'] = '3_frame'
multi['TP'] = count_df['true_and_3frame']
multi['FN'] = count_df['true']-count_df['true_and_3frame']
multi['FP'] = count_df['3frame']-count_df['true_and_3frame']

empty = count_df.filter(items=['file', 'frame'])
empty['model'] = 'empty'
empty['TP'] = count_df['true_and_empty']
empty['FN'] = count_df['true']-count_df['true_and_empty']
empty['FP'] = count_df['empty']-count_df['true_and_empty']

tp_df = pd.concat([single, multi, empty], ignore_index=True)

tp_df.head()
tp_df.to_csv('single_3frame_empty_count.csv') 

plt.figure(figsize=(20,10))
g3 = sns.boxplot(data=tp_df, x='file', y='recall', hue='model')

plt.figure(figsize=(20,10))
g3 = sns.boxplot(data=tp_df, x='file', y='average_precision', hue='model')

plt.figure(figsize=(5,10))
g3 = sns.violinplot(data=tp_df, x='model', y='FN')

tp_df['F1-score'] = 2*tp_df['precision']*tp_df['recall']/(tp_df['precision']+tp_df['recall'])

plt.figure(figsize=(20,10))
g3 = sns.boxplot(data=tp_df, x='file', y='F1-score', hue='model')

keras.backend.clear_session()
stride = 2

# #IOU
for stack in ogm3_data.keys():
    y_pred1 = None
    y_pred2 = None
    y_pred3 = None
    
    img_stack = ogm3_data[stack]
    for i in range(0, len(img_stack["image_patch"]), stride):
        pred_si = model_single.predict(np.expand_dims(img_stack["image_patch"][i:i+stride,1,:,:], axis=1))
        pred_3f = model_3frame.predict(img_stack["image_patch"][i:i+stride])
        pred_empt = model_empty.predict(img_stack["image_patch"][i:i+stride])
        if y_pred1 is not None:
            y_pred1 = np.concatenate((y_pred1, pred_si))
            y_pred2 = np.concatenate((y_pred2, pred_3f))
            y_pred3 = np.concatenate((y_pred3, pred_empt))

        if y_pred1 is None:
            y_pred1 = pred_si
            y_pred2 = pred_3f
            y_pred3 = pred_empt
    
    ogm3_data[stack]["y_pred_single"] = unpatch_stack(y_pred1, 8, 8, 1)
    ogm3_data[stack]["y_pred_3frame"] = unpatch_stack(y_pred2, 8, 8, 1)
    ogm3_data[stack]["y_pred_empty"] = unpatch_stack(y_pred3, 8, 8, 1)
    print(stack, ogm3_data[stack]["y_pred_single"].shape, ogm3_data[stack]["y_pred_3frame"].shape, ogm3_data[stack]["y_pred_empty"].shape)
    

threshold = 0.5

for stack in ogm3_data.keys():
    pred_si = (ogm3_data[stack]["y_pred_single"]>threshold)*255
    pred_3f = (ogm3_data[stack]["y_pred_3frame"]>threshold)*255
    pred_empty = (ogm3_data[stack]["y_pred_empty"]>threshold)*255
    
    saveme = np.concatenate((np.expand_dims(ogm3_data[stack]["y_true"],axis=1), pred_si, pred_3f, pred_empty), axis=1)
    saveme = saveme.astype('uint8')
    prefix="V4_compare"
    dic = unpatch_stack(ogm3_data[stack]["image_patch"], 8, 8, 3)
    dic = dic[:,1,:,:] * 255
    dic = np.expand_dims(dic, axis=1).astype('uint8')
    print(dic.shape, ogm3_data[stack]["image_patch"].max())
    saveme = np.concatenate((dic, saveme), axis=1)
    tiff.imwrite(os.path.join(r"C:\Users\Jens\Documents\Code\BactUnet\Bactnet\Training data\stacks\predict", "OGM3.tif"), saveme, imagej=True,
                      metadata={'unit': 'um', 'finterval': 15,
                                'axes': 'TCYX'})

                                
