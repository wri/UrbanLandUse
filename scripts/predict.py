import os,sys
PROJECT_DIR=os.environ['ULU_REPO']
sys.path.append(PROJECT_DIR)
import re
from datetime import datetime
import csv
import yaml
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
import tensorflow as tf
import tensorflow.keras.backend as K
import utils.batch_generator as batch_generator
import utils.util_network as util_network
import utils.util_training as util_training
import utils.util_scoring as util_scoring



def record_model_creation_8cat(
        model_id, notes, place_images, ground_truth, resolution, stack_label, feature_count, window, category_label,cats_map, balancing, 
        model_summary, epochs, batch_size,
        train_confusion, train_recalls, train_precisions, train_accuracy,
        train_f_score_open, train_f_score_nonres,train_f_score_resatomistic_informal,train_f_score_resatomistic_formal, train_f_score_resinformal,
        train_f_score_resformal,train_f_score_resproject, train_f_score_roads, train_f_score_average,
        valid_confusion, valid_recalls, valid_precisions, valid_accuracy,
        valid_f_score_open, valid_f_score_nonres, valid_f_score_resatomistic_informal,valid_f_score_resatomistic_formal, valid_f_score_resinformal,
        valid_f_score_resformal,valid_f_score_resproject, valid_f_score_roads, valid_f_score_average,
        datetime=datetime.now(),
        scorecard_file=SCORECARD_PATH):
    
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

"""
1. get train/valid gen

"""

section('PREDICTIONS')
train_gen.reset()
predictions_t = network.predict_generator(
    train_gen, 
    steps=train_gen.steps, 
    verbose=1,
    use_multiprocessing=True,
    max_queue_size=40,
    workers=64 )

valid_gen.reset()
predictions_v = network.predict_generator(
    valid_gen, 
    steps=valid_gen.steps, 
    verbose=1,
    use_multiprocessing=True,
    max_queue_size=40,
    workers=64 )

print(predictions_t.shape,predictions_v.shape)


# In[28]:


Yhat_t = predictions_t.argmax(axis=-1)
Yhat_v = predictions_v.argmax(axis=-1)
print(Yhat_t.shape,Yhat_v.shape)


# In[29]:


Y_t = train_gen.get_label_series().values
Y_v = valid_gen.get_label_series().values
print(Y_t.shape,Y_v.shape)


# In[30]:


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


# In[31]:


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


# In[32]:


train_recalls_expanded = [train_recalls[0],train_recalls[1],train_recalls[2],None,None,None,None,None]
valid_recalls_expanded = [valid_recalls[0],valid_recalls[1],valid_recalls[2],None,None,None,None,None]
train_precisions_expanded = [train_precisions[0],train_precisions[1],train_precisions[2],None,None,None,None,None]
valid_precisions_expanded = [valid_precisions[0],valid_precisions[1],valid_precisions[2],None,None,None,None,None]


# In[33]:


place_locales_filename = data_root+'models/'+model_id+'_place_locales.pkl'
category_weights_filename = data_root+'models/'+model_id+'_category_weights.pkl'
network_filename = data_root+'models/'+model_id+'.hd5'

if os.path.exists(network_filename):
    newpath=backup_path(network_filename)
    print('[WARNING] file already exists at ('+network_filename+') moving to ('+newpath+')')
    os.rename(network_filename,newpath)
if os.path.exists(category_weights_filename):
    newpath=backup_path(category_weights_filename)
    print('[WARNING] file already exists at ('+category_weights_filename+') moving to ('+newpath+')')
    os.rename(category_weights_filename,newpath)

print(network_filename)
pickle.dump(category_weights, open(category_weights_filename, 'wb'))
network.save(network_filename, include_optimizer=True)
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

