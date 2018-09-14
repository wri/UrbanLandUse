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
#import pickle
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
        print "input shapes", Yhat.shape, Y.shape
        print "scores [tp, fp, fn, tn]", tp, fp, fn, tn, \
        	"accuracy", round(100*((tp + tn)/float(tp + fp + fn + tn)),1),"%"

def calc_confusion(Yhat,Y,categories):
    n_categories = len(categories)
    confusion = np.zeros((n_categories,n_categories),dtype='uint32')
    for j in range(n_categories):
        for i in range(n_categories):
            confusion[j,i] = np.sum(np.logical_and((Yhat==categories[i]),(Y==categories[j])))
            # print j,i, confusion[j,i], categories[j], categories[i] 
        print categories[j], np.sum(confusion[j,:])
    print confusion
    print confusion.sum(), confusion.trace(), confusion.trace()/float(confusion.sum())
    return confusion

def calc_confusion_detail(Yhat,Y,categories):
    n_categories = len(categories)
    confusion = np.zeros((n_categories,n_categories),dtype='uint32')
    for j in range(n_categories):
        for i in range(n_categories):
            confusion[j,i] = np.sum(np.logical_and((Yhat==categories[i]),(Y==categories[j])))
            # print j,i, confusion[j,i], categories[j], categories[i] 
        print categories[j], np.sum(confusion[j,:])
    conf_str = ''
    end_str = ''
    for j in range(n_categories):
        for i in range(n_categories):
            conf_str += str(confusion[j,i]) + '\t'
        #conf_str += str(confusion[j]) + ' '
        conf_str += str(round(float(confusion[j,j]) / float(np.sum(confusion[j,:])),3)) + '\n'
        end_str +=  str(round(float(confusion[j,j]) / float(np.sum(confusion[:,j])),3)) + '\t'
    conf_str += end_str
    #print np.array_str(confusion)
    print conf_str
    
    print confusion.sum(), confusion.trace(), confusion.trace()/float(confusion.sum())
    return confusion

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
    print X_train_scaled.shape, X_valid_scaled.shape
    # print X_train_scaled[19,:]
    return X_train_scaled, X_valid_scaled, scaler
    
    
def train_model_svm(X_train_scaled, X_valid_scaled, Y_train, Y_valid, categories=[0,1,4,6], alpha=0.003,penalty='l2'):
    model = SGDClassifier(alpha=alpha,penalty=penalty) 
    model.fit(X_train_scaled, Y_train)
    ## evaluate model
    print "evaluate training"
    Yhat_train = model.predict(X_train_scaled)
    conf = calc_confusion(Yhat_train,Y_train,categories)
    print "evaluate validation"
    Yhat_valid = model.predict(X_valid_scaled)
    conf = calc_confusion(Yhat_valid,Y_valid,categories)
    return Yhat_train, Yhat_valid, model
