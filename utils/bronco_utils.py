from __future__ import print_function
import cartopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
import seaborn as sn
import pandas as pd


#
# DECARTES
#
def place_slug(*parts):
    return "_".join(parts)


#
# IMAGE INFO
#
def maxmin_info(img):
    mns=np.min(img,axis=(0,1))
    mxs=np.max(img,axis=(0,1))
    print('band, min, max')
    for i,(mn,mx) in enumerate(zip(mns,mxs)):
        print(i,mn,mx)


#
# PLOTS
#
def plot_image(array,figsize=(6,6)):
    plt.imshow(array)
    plt.show()


def plot_place(pl,figsize=(6,6)):
    albers = cartopy.crs.AlbersEqualArea(
        central_latitude=pl.lat, central_longitude=pl.lon)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=albers)
    crs=cartopy.crs.PlateCarree()
    ax.add_geometries([pl.shape],crs)
    ax.set_extent(
        (pl.bounds[0], pl.bounds[2], pl.bounds[1], pl.bounds[3]), 
        crs=crs)
    ax.gridlines(crs=crs)
    plt.show()


def plot_confusion(categories,labels,trues,preds,title=None):
    cm=confusion_matrix(trues,preds,categories)
    df_cm=pd.DataFrame(cm,labels,labels)
    ax=plt.axes()
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm,annot=True,annot_kws={"size": 16},fmt='g',ax=ax)
    if title: ax.set_title(title)
    plt.show()

    
def print_metrics(trues,preds,title=None):
    if title: print("\n{}:".format(title))
    print("\taccuracy: ",accuracy_score(trues,preds))
    print("\tprecision: ",precision_score(trues,preds,average='macro'))
    print("\trecall: ",recall_score(trues,preds,average='macro'))
    print("\tf2: ",fbeta_score(trues,preds,2,average='macro'))

    
def print_results(
        categories,
        labels,
        trues,
        preds,
        trues_valid=None,
        preds_valid=None):
    print_metrics(trues,preds,"TRAINING")
    plot_confusion(categories,labels,trues,preds,"CONFUSION MATRIX: TRAINING")
    if not ((trues_valid is None) or (preds_valid is None)):
        print("\n\n---------------------------------------\n\n")
        print_metrics(trues_valid,preds_valid,"VALIDATION")
        plot_confusion(
            categories,
            labels,
            trues_valid,
            preds_valid,
            "CONFUSION MATRIX: VALIDATION")






