# organized set of helper functions
# drawn from bronco.py and bronco notebooks
# topic: machine learning

import warnings
warnings.filterwarnings('ignore')
#
#import os
#import sys
#import json
#import itertools
import pickle
import pandas as pd
#from pprint import pprint
#
import numpy as np
#import shapely
#import cartopy
from osgeo import gdal
#import matplotlib.pyplot as plt
#
#import descarteslabs as dl
import sklearn
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDClassifier



# DISPLAY

# going forward, swap this out in favor of more mature scikit scoring
def score(Yhat,Y,report=True):
    totals = np.zeros(4,dtype='uint16')
    tp = np.logical_and((Yhat==1), (Y==1)).sum()
    fp = np.logical_and((Yhat==0), (Y==1)).sum()
    fn = np.logical_and((Yhat==1), (Y==0)).sum()
    tn = np.logical_and((Yhat==0), (Y==0)).sum()
    totals[0] += tp
    totals[1] += fp
    totals[2] += fn
    totals[3] += tn
    if report==True:
        print("input shapes", Yhat.shape, Y.shape)
        print("scores [tp, fp, fn, tn]", tp, fp, fn, tn, \
            "accuracy", round(100*((tp + tn)/float(tp + fp + fn + tn)),1),"%")



# think this is stuff from brookie; presumably confusion_matrix function is from sklearn
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


# PREPROCESSING & TRAINING

def scale_learning_data(X_train, X_valid):
    scaler = StandardScaler()
    scaler.fit(X_train)
    # apply scaler
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    print(X_train_scaled.shape, X_valid_scaled.shape)
    # print(X_train_scaled[19,:])
    return X_train_scaled, X_valid_scaled, scaler
    
    
def train_model_svm(X_train_scaled, X_valid_scaled, Y_train, Y_valid, categories=[0,1,4,6], alpha=0.003,penalty='l2'):
    model = SGDClassifier(alpha=alpha,penalty=penalty) 
    model.fit(X_train_scaled, Y_train)
    ## evaluate model
    print("evaluate training")
    Yhat_train = model.predict(X_train_scaled)
    conf = calc_confusion(Yhat_train,Y_train,categories)
    print("evaluate validation")
    Yhat_valid = model.predict(X_valid_scaled)
    conf = calc_confusion(Yhat_valid,Y_valid,categories)
    return Yhat_train, Yhat_valid, model


# from class weightings development

def load_training_data(city, suffix, label_suffix, stack_label, window,data_root, resolution=10, typ='train'):

    train_file = data_root+city+'/'+city+'_'+typ+'_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
    with open(train_file, 'rb') as f:
        X_train, Y_train = pickle.load(f)
    return X_train, Y_train

def get_category_counts(place_images,category_label,label_suffix,stack_label,window,data_root,resolution=10):
    image_names=[]
    category_counts={category_label[c]: [] for c in range(7) }
    for city, suffixes in place_images.items():
        for suffix in suffixes:
            image_names.append("{}_{}".format(city,suffix))
            _,Y_train=load_training_data(city,suffix,label_suffix, stack_label, window,data_root,resolution=resolution)
            # can insert remapping here if necessary,
            # then calculate counts off of modified/remapped data
            categories,counts=np.unique(Y_train,return_counts=True)
            for c,cnt in zip(categories,counts):
                category_counts[category_label[c]].append(cnt)
    df=pd.DataFrame()
    df['image_name']=image_names
    for cat,cnt in category_counts.items():
        df[cat]=cnt
    return df

def get_category_counts_simple(Y_train):
    df=pd.DataFrame()
    categories,counts=np.unique(Y_train,return_counts=True)
    df = pd.DataFrame(counts.reshape(1,2))
    return df

def normalize_weights(weights,max_score=None):
    mx=min(weights)
    weights=[ w/mx for w in weights ]
    if max_score:
        weights=[ min(max_score,w) for w in weights ]
    return weights


def category_weights(category_counts,mu=1.0,use_log=False,max_score=None):
    weights=[]
    total=np.sum(category_counts)
    for count in category_counts:
        if not count:
            count=EPS
        score=total/float(count)
        if use_log:
            score = math.log(mu*score)
        weights.append(score)
    return normalize_weights(weights,max_score)

def image_names(place_images):
    names=[]
    for city, suffixes in place_images.items():
        for suffix in suffixes:
            names.append("{}_{}".format(city,suffix))
    return names    

def generate_category_weights(place_images,category_label,label_suffix,stack_label,window,data_root,
        use_log=True, columns=['image_name','Open Space','Non-Residential','Residential-Total','Roads'],
        resolution=10):
    df = get_category_counts(place_images,category_label,label_suffix,stack_label,window,data_root,resolution=resolution)
    df['Residential-Total']=df['Residential Atomistic']+df['Residential Informal Subdivision']+df['Residential Formal Subdivision']+df['Residential Housing Project']
    COLUMNS=columns
    df=df[COLUMNS]
    labels=COLUMNS[1:]
    cat_counts=[df.sum()[l] for l in labels]
    weights=category_weights(cat_counts,use_log)
    return weights

def generate_category_weights_simple(Y_train,use_log=True):
    df = get_category_counts_simple(Y_train)
    print(df.head())
    cat_counts=[df.sum()[l] for l in range(2)]
    weights=category_weights(cat_counts,use_log)
    return weights

def make_binary(Y, category, silent=True):
    # create binary data
    Y_bin = Y.copy()
    cat_mask = (Y==category)
    Y_bin[cat_mask==1] = 1
    Y_bin[cat_mask==0] = 0
    if not silent:
        print(Y[0:20])
        print(Y_bin[0:20])
    return Y_bin

def balance_binary(X, Y, max_ratio=3.0, silent=True):
    where_false = np.where(Y==0)
    where_true = np.where(Y==1)

    n_false = len(where_false[0])
    n_true = len(where_true[0])

    n_array = np.array([n_false,n_true])
    arg_min = np.argmin(n_array)
    arg_max = np.argmax(n_array)
    n_min = n_array[arg_min]
    n_max = n_array[arg_max]

    if not silent:
        print('min:', n_min, '; max:', n_max)
    if (int(max_ratio * n_min) > n_max):
        # no balancing necessary; proportions already within acceptable range
        return X, Y

    n_special = int(max_ratio * n_min)
    n_balanced = n_min + n_special

    X_balanced_shape = (n_balanced,)+ X.shape[1:]
    if not silent: 
        print(X_balanced_shape)

    X_balanced = np.zeros(X_balanced_shape,dtype=X.dtype)
    Y_balanced = np.zeros((n_balanced),dtype=Y.dtype)

    perm_false = np.random.permutation(n_false)
    perm_true = np.random.permutation(n_true)

    false_samples = where_false[0][perm_false[:]]
    true_samples = where_true[0][perm_true[:]]

    if arg_min==0: # more true samples than false
        X_balanced[:n_min] = X[(false_samples,)]
        Y_balanced[:n_min] = Y[(false_samples,)]
        X_balanced[n_min:] = X[(true_samples[:n_special],)]
        Y_balanced[n_min:] = Y[(true_samples[:n_special],)]
    else:
        X_balanced[:n_min] = X[(true_samples,)]
        Y_balanced[:n_min] = Y[(true_samples,)]
        X_balanced[n_min:] = X[(false_samples[:n_special],)]
        Y_balanced[n_min:] = Y[(false_samples[:n_special],)]

    perm_balanced = np.random.permutation(n_balanced)
    X_balanced = X_balanced[perm_balanced]
    Y_balanced = Y_balanced[perm_balanced]

    if not silent:
        print(np.sum(Y_balanced==0),np.sum(Y_balanced!=0))
        print(Y_balanced.shape)
        print(Y_balanced[0:20])

    return X_balanced, Y_balanced

