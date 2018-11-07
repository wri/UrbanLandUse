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

def calc_confusion_details(confusion):
    n_categories = confusion.shape[0]
    # out of samples in category, how many assigned to that category
    # (true positives) / (true positives + false negatives)
    # (correct) / (samples from category)
    recalls = np.zeros(n_categories, dtype='float32')
    # out of samples assigned to category, how many belong to that category
    # (true positives) / (true positives + false positives)
    # (correct) / (samples assigned to category)
    precisions = np.zeros(n_categories, dtype='float32')

    for j in range(n_categories):
        ascribed = np.sum(confusion[:,j])
        actual = np.sum(confusion[j,:])
        correct = confusion[j,j]
        recalls[j] = float(correct)/float(actual)
        precisions[j] = float(correct)/float(ascribed)

    # what percentage of total samples were assigned to the correct category
    accuracy = confusion.trace()/float(confusion.sum())

    return recalls, precisions, accuracy


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


# KERAS BUILDING BLOCKS

def pool_dropout(x):
    x=MaxPooling2D(pool_size=(2, 2))(x)
    return Dropout(0.25)(x)

def denselayers(x,output_nodes=4):
    x=Dense(512)(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(output_nodes)(x)
    return Activation('softmax')(x)

def conv_block(filters,x,kernel_size=3):
    x=Conv2D(filters, kernel_size, padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(filters, kernel_size, padding='same')(x)
    x=Activation('relu')(x)
    return x

def resnet_block(filters,x,weights=None):
    x_3=conv_block(filters,x)
    x_1=Conv2D(filters, (1, 1))(x)
    if weights:
        x_3=Lambda(lambda y: y*weights[0])(x_3)
        x_1=Lambda(lambda y: y*weights[1])(x_1)
    x=keras.layers.Add()([x_3,x_1])
    return x

def doubleres_block(filters,x,weights=[0.9,1.0,1.1]):
    x_5=conv_block(filters,x, kernel_size=5)
    x_3=conv_block(filters,x)
    x_1=Conv2D(filters, (1, 1))(x)
    if weights:
        x_5=Lambda(lambda y: y*weights[0])(x_5)
        x_3=Lambda(lambda y: y*weights[1])(x_3)
        x_1=Lambda(lambda y: y*weights[2])(x_1)
    x=keras.layers.Add()([x_5,x_3,x_1])
    return x


def build_model(cblock,filters1,filters2,print_summary=True,input_shape=(8,17,17)):
    inputs=Input(shape=input_shape) 
    x=cblock(filters1,inputs)
    x=pool_dropout(x)
    x=cblock(filters2,x)
    x=pool_dropout(x)
    x=Flatten()(x)
    x=denselayers(x)
    m=Model(inputs=inputs, outputs=x)
    if print_summary:
        m.summary()
    return m

