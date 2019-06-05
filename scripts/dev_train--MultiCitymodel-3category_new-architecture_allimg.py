#!/usr/bin/env python
# coding: utf-8

# # Development: Train 3-Category Classifier
# Use the latest and greatest model structures, parameters, workflow (ie class weighting, learning fast & slow, etc) to train a 3-category classifier using only the non-road training samples.
# 
# This is the latest version as of the start of Phase IV, meant to leverage the previous changes in order to utilize the `fit_generator`.
# 
# Date: 2019-02-01  
# Author: Peter Kerins  

# ## Preparation

# ### Import statements
# (may be over-inclusive)

# In[1]:


# typical, comprehensive imports
import warnings
warnings.filterwarnings('ignore')
#
import os
import sys
import json
import itertools
import pickle
from pprint import pprint
#
import numpy as np
import geojson
import fiona
import gdal
import h5py
# get_ipython().magic(u'matplotlib inline')
# import matplotlib.pyplot as plt

import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler 
import ogr, gdal
from tensorflow.keras.models import load_model
import math
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Add, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History

import collections
from pprint import pprint

import descarteslabs as dl
# print dl.places.find('illinois') ## TEST

ULU_REPO = os.environ["ULU_REPO"]
sys.path.append(ULU_REPO+'/utils')
print(sys.path)

import util_descartes
import util_ml
import util_rasters
import util_vectors
import util_workflow
import util_chips
import util_training
import util_network
import util_scoring
from batch_generator import BatchGenerator


# ### Set key variables

# In[2]:


data_root='/data/phase_iv/'

tile_resolution = 5
tile_size = 256
tile_pad = 32
resolution=tile_resolution  # Lx:15 S2:10

processing_level = None
source = 's2'
#image_suffix = 'E'

s2_bands=['blue','green','red','nir','swir1','swir2','alpha']; s2_suffix='BGRNS1S2A'  # S2, Lx

s1_bands=['vv','vh']; s1_suffix='VVVH'  

resampling='bilinear'
processing = None

label_suffix = 'aue'
label_lot = '0'

# can do much more sophisticated filtering than this, but fine for demonstration
place_images = {}
place_images['hindupur']=['U', 'V', 'W', 'X', 'Y', 'Z']
place_images['singrauli']=['O','P','Q','R','S','T','U']
place_images['vijayawada']=['H','I']
place_images['jaipur']=['T','U','W','X','Y','Z']
place_images['hyderabad']=['P','Q','R','S','T','U']
place_images['sitapur']=['Q','R','T','U','V']
place_images['kanpur']=['AH', 'AK', 'AL', 'AM', 'AN']
place_images['belgaum']=['P','Q','R','S','T']
place_images['parbhani']=['T','V','W','X','Y','Z']
place_images['pune']=['P', 'Q', 'T', 'U', 'S']
place_images['ahmedabad']= ['Z', 'V', 'W', 'X', 'Y', 'AA']
place_images['malegaon']=  ['V', 'W', 'X', 'Y', 'Z']
place_images['kolkata'] =  ['M','N','O','P','Q','R']
place_images['mumbai']=['P','Q','R','S','U','V']


# In[3]:


category_label = {0:'Open Space',1:'Non-Residential',                   2:'Residential Atomistic',3:'Residential Informal Subdivision',                   4:'Residential Formal Subdivision',5:'Residential Housing Project',                   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'}

cats_map = {}
cats_map[0] = 0
cats_map[1] = 1
cats_map[2] = 2
cats_map[3] = 2
cats_map[4] = 2
cats_map[5] = 3


# ### Specify training data & training parameters

# In[4]:


window = 17

# bands stuff outdated! needs to be reconciled with catalog filtering
# will ignore for the moment since this is a bigger fix...
# haven't done any examples yet incorporating additional chips beyond s2
# into construction of a training sample
bands_vir=s2_bands[:-1]
bands_sar=None
bands_ndvi=None
bands_ndbi=None
bands_osm=None

# this can get updated when cloudmasking is added
haze_removal = False

epochs = 500 # this is fine, if irrelevant
batch_size = 128
balancing = None


# In[5]:


# needs to be updated completely; bands stuff doesn't make sense right now
stack_label, feature_count = util_workflow.build_stack_label(
        bands_vir=bands_vir,
        bands_sar=bands_sar,
        bands_ndvi=bands_ndvi,
        bands_ndbi=bands_ndbi,
        bands_osm=bands_osm,)
print(stack_label, feature_count)

""" CREATE DATASET

# ### Specify data of interest
# Load and manipulate the catalog to specify which samples are of interest for this training.

# In[6]:

df = util_chips.load_catalog()
print(len(df.index))


# In[7]:


mask = pd.Series(data=np.zeros(len(df.index),dtype='uint8'), index=range(len(df)), dtype='uint8')

for place,image_list in place_images.items():
    for image in image_list:
        mask |= (df['city']==place) & (df['image']==image)

# straight away remove road samples
mask &= (df['lulc']!=6)

# filter others according to specifications
mask &= (df['gt_type']==label_suffix)
mask &= (df['gt_lot']==int(label_lot))
mask &= (df['source']==source)
mask &= (df['resolution']==int(resolution))
mask &= (df['resampling']==resampling)
mask &= (df['processing']==str(processing).lower())

print(np.sum(mask))


# In[8]:


#here for example we will just exclude all roads samples
df = df[mask]
df.reset_index(drop=True,inplace=True)
len(df)


"""


""" LOAD DATASET FROM CSV 
DSET_PATH='all_city_dataset.csv'
df=pd.read_csv(DSET_PATH)
"""


""" CREATE TRAIN/VALID SETS 

# In[9]:


place_locales_paths = ['/data/phase_iv/models/3cat_Ahm_V-AA_place_locales.pkl',
                       '/data/phase_iv/models/3cat_Bel_P-T_place_locales.pkl'       ,
                       '/data/phase_iv/models/3cat_Hin_U-Z_place_locales.pkl'       ,
                       '/data/phase_iv/models/3cat_Hyd_P-U_place_locales.pkl'       ,
                       '/data/phase_iv/models/3cat_Jai_T-U+W-Z_place_locales.pkl'   ,
                       '/data/phase_iv/models/3cat_Kan_AH+AK-AN_place_locales.pkl'  ,
                       '/data/phase_iv/models/3cat_Mal_V-Z_place_locales.pkl'       ,
                       '/data/phase_iv/models/3cat_Par_T+V-Z_place_locales.pkl',
                       '/data/phase_iv/models/3cat_Pun_P-Q+S-U_place_locales.pkl',
                       '/data/phase_iv/models/3cat_Sin_O-U_place_locales.pkl',
                       '/data/phase_iv/models/3cat_Sit_Q-R+T-V_place_locales.pkl',
                       '/data/phase_iv/models/3cat_Vij_H-I_place_locales.pkl',
                       '/data/phase_iv/models/3cat_Kol_M-R_place_locales.pkl',
                       '/data/phase_iv/models/3cat_Mum_P-V_place_locales.pkl'
                       ]


# In[10]:


combined_place_locales = {}
print(combined_place_locales)
for place_locales_filename in place_locales_paths:
    with open(place_locales_filename, "rb") as f:
        place_locales = pickle.load(f,encoding='latin1')
    combined_place_locales.update(place_locales)
print(combined_place_locales)


# In[11]:


df_t, df_v = util_chips.mask_locales(df, combined_place_locales)
print(len(df_t), len(df_v))

"""


""" LOAD TRAIN/VALID DATASETS FROM CSV """
df_t=pd.read_csv('all_city_dataset-train.csv')
df_v=pd.read_csv('all_city_dataset-valid.csv')
# df=pd.concat([df_t,df_v])

# ## Training
# ### Build loss function
# #### Inspect dataframes for consistency
# In[12]:


print(util_training.calc_category_counts(df_t,remapping=None), len(df_t))
print(util_training.calc_category_counts(df_v,remapping=None), len(df_v))
# print()
# print(util_training.calc_category_counts(df,remapping=None), len(df))


# #### Generate class weighting information

# In[13]:


# category_weights =  util_training.generate_category_weights(df,remapping='standard',log=False,mu=1.0,max_score=None)
category_weights = {0: 1.231944246656025, 1: 1.8600266479274625, 2: 1.0}
# del(df)

# In[ ]:


print(category_weights.items())
weights = list(zip(*category_weights.items()))[1]
print(weights)


# #### Use weights to make weighted categorical crossentropy loss function

# In[ ]:


loss = util_training.make_loss_function_wcc(weights)


# ### Build convolutional neural network

# #### Name and briefly describe model

# In[ ]:


model_id = '3cat_14ct_green_2017_all_img'
notes = 'training a 14 city multi city model using all the img for each city from the remaining Natgeo imagery from each cities, 3cat 5m bilinear, no processing, locale based sample distribution'


# In[ ]:


print("DEFAULT:",K.image_data_format())
K.set_image_data_format('channels_first')
print("UPDATED:",K.image_data_format())


# #### Create actual model

# In[ ]:


# #hardcoded params
# network=util_network.build_model(util_network.doubleres_block,input_shape=(6,17,17),output_nodes=3)
# util_network.compile_network(network, loss, LR=0.001)

#hardcoded params
network=util_network.build_xmodel(input_shape=(6,17,17),output_nodes=3,input_conv_block=True)
util_network.compile_network(network, loss, LR=0.001)


# ---

# ### Train model fast

# #### Create generators to provide training samples

# In[ ]:


generator_t = BatchGenerator(df_t,remapping='3cat',look_window=window,batch_size=batch_size,one_hot=3)
generator_v = BatchGenerator(df_v,remapping='3cat',look_window=window,batch_size=batch_size,one_hot=3)


# #### Create callback functions for training

# In[ ]:


weights_label='weights_fast'
callbacks, weights_path = util_training.create_callbacks(data_root, model_id, weights_label=weights_label, patience=2)


# #### Conduct training using `fit_generator` and visualize progress

# In[ ]:


# train fast
#history_fast = network.fit(X_train, Y_t_cat, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_v_cat), shuffle=True,callbacks=callbacks)
#docs: fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None,
                    #class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
history_fast = network.fit_generator(generator_t, epochs=epochs, callbacks=callbacks, steps_per_epoch=generator_t.steps,
                                    validation_data=generator_v, validation_steps=generator_v.steps,
                                    shuffle=True,use_multiprocessing=True,max_queue_size=40,workers=64,)

# plt.plot(history_fast.history['val_acc'])
# plt.show()
# plt.plot(history_fast.history['val_loss'])
# plt.show()


# ---

# ### Rebuild model and train slow

# In[ ]:


# #hardcoded params
# network=util_network.build_model(util_network.doubleres_block,input_shape=(6,17,17),output_nodes=3)
# # load weights from fast learning
# network.load_weights(weights_path)

# util_network.compile_network(network, loss, LR=0.0001)

#hardcoded params
network=util_network.build_xmodel(input_shape=(6,17,17),output_nodes=3,input_conv_block=True)
# load weights from fast learning
network.load_weights(weights_path)

util_network.compile_network(network, loss, LR=0.0001)


# #### Create generators to provide training samples

# In[ ]:


generator_t.reset()
generator_v.reset()


# #### Create callback functions for training

# In[ ]:


weights_label='weights_slow'
callbacks, weights_path = util_training.create_callbacks(data_root, model_id, weights_label=weights_label, patience=5)


# #### Conduct training using `fit_generator` and visualize progress

# In[ ]:


history_slow = network.fit_generator(generator_t, epochs=epochs, callbacks=callbacks, steps_per_epoch=generator_t.steps,
                                    validation_data=generator_v, validation_steps=generator_v.steps,
                                    shuffle=True,use_multiprocessing=True,max_queue_size=40,workers=64,)

# plt.plot(history_slow.history['val_acc'])
# plt.show()
# plt.plot(history_slow.history['val_loss'])
# plt.show()


# ---

# ## Scoring

# ### Apply model to training and validation data

# In[ ]:


generator_t.reset()
#predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
predictions_t = network.predict_generator(generator_t, steps=generator_t.steps, verbose=1,
                  use_multiprocessing=True,max_queue_size=40,workers=64,)
print(predictions_t.shape)

generator_v.reset()
#predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
predictions_v = network.predict_generator(generator_v, steps=generator_v.steps, verbose=1,
                  use_multiprocessing=True,max_queue_size=40,workers=64,)
print(predictions_v.shape)


# In[ ]:


Yhat_t = predictions_t.argmax(axis=-1)
print(Yhat_t.shape)
Yhat_v = predictions_v.argmax(axis=-1)
print(Yhat_v.shape)


# ### Extract corresponding _actual_ ground-truth values directly from catalog

# In[ ]:


Y_t = generator_t.get_label_series().values
print(Y_t.shape)
Y_v = generator_v.get_label_series().values
print(Y_v.shape)


# ### Generate typical scoring information

# In[ ]:


print("evaluate training")
# hardcoded categories
categories=[0,1,2]
train_confusion = util_scoring.calc_confusion(Yhat_t,Y_t,categories)
train_recalls, train_precisions, train_accuracy = util_scoring.calc_confusion_details(train_confusion)

# Calculate f-score
beta = 2
train_f_score = (beta**2 + 1) * train_precisions * train_recalls / ( (beta**2 * train_precisions) + train_recalls )
train_f_score_open = train_f_score[0] 
train_f_score_nonres = train_f_score[1]  
train_f_score_resatomistic_informal = train_f_score[2]
train_f_score_resatomistic_formal = None
train_f_score_resinformal = None  
train_f_score_resformal = None
train_f_score_resproject = None
train_f_score_roads = None
train_f_score_average = np.mean(train_f_score)


# In[ ]:


print("evaluate validation")
valid_confusion = util_scoring.calc_confusion(Yhat_v,Y_v,categories)
valid_recalls, valid_precisions, valid_accuracy = util_scoring.calc_confusion_details(valid_confusion)

# Calculate f-score
valid_f_score = (beta**2 + 1) * valid_precisions * valid_recalls / ( (beta**2 * valid_precisions) + valid_recalls )
valid_f_score_open = valid_f_score[0] 
valid_f_score_nonres = valid_f_score[1] 
valid_f_score_resatomistic_informal = valid_f_score[2]
valid_f_score_resatomistic_formal = None
valid_f_score_resinformal = None 
valid_f_score_resformal = None
valid_f_score_resproject = None
valid_f_score_roads = None
valid_f_score_average = np.mean(valid_f_score)


# In[ ]:


# expanding lists to match expected model_record stuff
train_recalls_expanded = [train_recalls[0],train_recalls[1],train_recalls[2],None,None,None,None,None]
valid_recalls_expanded = [valid_recalls[0],valid_recalls[1],valid_recalls[2],None,None,None,None,None]
train_precisions_expanded = [train_precisions[0],train_precisions[1],train_precisions[2],None,None,None,None,None]
valid_precisions_expanded = [valid_precisions[0],valid_precisions[1],valid_precisions[2],None,None,None,None,None]


# In[ ]:


import numpy as np
import util_rasters
import datetime
import csv


def record_model_creation_8cat(
        model_id, notes, place_images, ground_truth, resolution, stack_label, feature_count, window, category_label,cats_map, balancing, 
        model_summary, epochs, batch_size,
        train_confusion, train_recalls, train_precisions, train_accuracy,
        train_f_score_open, train_f_score_nonres,train_f_score_resatomistic_informal,train_f_score_resatomistic_formal, train_f_score_resinformal,
        train_f_score_resformal,train_f_score_resproject, train_f_score_roads, train_f_score_average,
        valid_confusion, valid_recalls, valid_precisions, valid_accuracy,
        valid_f_score_open, valid_f_score_nonres, valid_f_score_resatomistic_informal,valid_f_score_resatomistic_formal, valid_f_score_resinformal,
        valid_f_score_resformal,valid_f_score_resproject, valid_f_score_roads, valid_f_score_average,
        datetime=datetime.datetime.now(),
        scorecard_file='/data/phase_iv/models/scorecard_phase_iv_models.csv'):
    
    with open(scorecard_file, mode='a') as scorecard:
        score_writer = csv.writer(scorecard, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        score_writer.writerow([
            model_id, notes, datetime, place_images, ground_truth, resolution, stack_label, feature_count, window, category_label,cats_map, balancing, 
            model_summary, epochs, batch_size,
            train_confusion,train_recalls[0],train_recalls[1],train_recalls[2],train_recalls[3],train_recalls[4],train_recalls[5],train_recalls[6],train_recalls[7],
            train_precisions[0],train_precisions[1],train_precisions[2],train_precisions[3],train_precisions[4],train_precisions[5],train_precisions[6],train_precisions[7], 
            train_accuracy, train_f_score_open, train_f_score_nonres, train_f_score_resatomistic_informal,train_f_score_resatomistic_formal, train_f_score_resinformal,
            train_f_score_resformal,train_f_score_resproject, train_f_score_roads, train_f_score_average,
            valid_confusion, valid_recalls[0],valid_recalls[1],valid_recalls[2],valid_recalls[3],valid_recalls[4],valid_recalls[5],valid_recalls[6],valid_recalls[7], 
            valid_precisions[0],valid_precisions[1],valid_precisions[2],valid_precisions[3],valid_precisions[4],valid_precisions[5],valid_precisions[6],valid_precisions[7],
            valid_accuracy,valid_f_score_open, valid_f_score_nonres,valid_f_score_resatomistic_informal,valid_f_score_resatomistic_formal, valid_f_score_resinformal,
            valid_f_score_resformal,valid_f_score_resproject, valid_f_score_roads, valid_f_score_average,
            ])
    print('model scorecard updated')
    return


# ### Record experiment configuration and results

# In[ ]:


reload(util_scoring)
#scaler_filename = data_root+'models/'+model_id+'_scaler.pkl'
#model_filename  = data_root+'models/'+model_id+'_SVM.pkl'
place_locales_filename = data_root+'models/'+model_id+'_place_locales.pkl'
category_weights_filename = data_root+'models/'+model_id+'_category_weights.pkl'
network_filename = data_root+'models/'+model_id+'.hd5'

if os.path.exists(network_filename):
    print('Aborting all pickle operations: file already exists at specified path ('+network_filename+')')
elif os.path.exists(category_weights_filename):
    print('Aborting all pickle operations: file already exists at specified path ('+category_weights_filename+')')
else:
    print(network_filename)
    pickle.dump(place_locales, open(place_locales_filename, 'wb'))
    pickle.dump(category_weights, open(category_weights_filename, 'wb'))
    network.save(network_filename)
    # tracking only occurs if all saves are successful
    record_model_creation_8cat(
        model_id, notes, place_images, label_suffix+label_lot, resolution, stack_label, feature_count, 
        window,  category_label,cats_map, balancing, 
        network.get_config(), epochs, batch_size,
        train_confusion, train_recalls_expanded, train_precisions_expanded, train_accuracy,
        train_f_score_open, train_f_score_nonres, train_f_score_resatomistic_informal,train_f_score_resatomistic_formal, 
        train_f_score_resinformal,train_f_score_resformal,train_f_score_resproject, train_f_score_roads, train_f_score_average,
        valid_confusion, valid_recalls_expanded, valid_precisions_expanded, valid_accuracy,
        valid_f_score_open, valid_f_score_nonres, valid_f_score_resatomistic_informal,valid_f_score_resatomistic_formal,
        valid_f_score_resinformal,valid_f_score_resformal,valid_f_score_resproject, valid_f_score_roads, valid_f_score_average,)


# ---
