#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
PROJECT_DIR=os.environ['ULU_REPO']
sys.path.append(PROJECT_DIR)
print(PROJECT_DIR)


# In[2]:


import re
from datetime import datetime
import csv
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
print(tf.__version__)


# In[3]:


import utils.batch_generator as batch_generator
import utils.util_network as util_network
import utils.util_training as util_training
import utils.util_scoring as util_scoring





# In[5]:


REMAPPING='3cat'
NB_CATS=3
BANDS_LAST=True
DATASETS_DIR=f'{PROJECT_DIR}/datasets'
TRAIN_DS=f'{DATASETS_DIR}/multi_city_2img-train.csv'
VALID_DS=f'{DATASETS_DIR}/multi_city_2img-valid.csv'
SCORECARD_PATH='/data/phase_iv/models/scorecard_phase_iv_models.csv'
TS_FMT="%Y%m%d-%H:%M:%S"


# In[7]:
def section(sec):
    print('\n'*4)
    print('-'*100)
    print(sec.upper())
    print('-'*100)
    print('\n'*1)


def backup_path(path):
    parts=path.split('.')
    base='.'.join(parts[:-1])
    ext=parts[-1]
    return f'{base}.{datetime.now().strftime(TS_FMT)}.bak.{ext}'
    
    
def model_name(dsname,typ):
    dsname=os.path.basename(dsname)
    dsname=re.sub('.csv','',dsname)
    dsname=re.sub('-train','',dsname)
    return f'{dsname}-{typ}.h5'


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



# In[8]:


section('SETUP')


if BANDS_LAST:
    INPUT_SHAPE=(17,17,6)
    K.set_image_data_format('channels_last')
else:
    INPUT_SHAPE=(6,17,17)
    K.set_image_data_format('channels_first')


# In[6]:

print("BANDS_LAST:",BANDS_LAST,"DATA-FORMAT:",K.image_data_format())


data_root='/data/phase_iv/'
model_id = '3cat_14ct_green_2017_2-img-bl'
notes = (
    'training a 14 city multi city model '
    'using two img from each cities, '
    '3cat 5m bilinear, '
    'no processing, '
    'locale based sample distribution' )


# In[9]:


window = 17
batch_size=128

balancing = None
feature_count=6
stack_label='vir'

tile_resolution = 5
tile_size = 256
tile_pad = 32
resolution=tile_resolution  # Lx:15 S2:10

processing_level = None
source = 's2'

s2_bands=['blue','green','red','nir','swir1','swir2','alpha']; s2_suffix='BGRNS1S2A'  # S2, Lx

s1_bands=['vv','vh']; s1_suffix='VVVH'  

resampling='bilinear'
processing = None


# In[10]:


label_suffix = 'aue'
label_lot = '0'

place_images = {}
place_images['hindupur']=['U', 'V']
place_images['singrauli']=['O','P']
place_images['vijayawada']=['H','I']
place_images['jaipur']=['T','U']
place_images['hyderabad']=['P','Q']
place_images['sitapur']=['Q','R']
place_images['kanpur']=['AH', 'AK']
place_images['belgaum']=['P','Q']
place_images['parbhani']=['T','V']
place_images['pune']=['P', 'Q']
place_images['ahmedabad']= ['Z', 'V']
place_images['malegaon']=  ['V', 'W']
place_images['kolkata'] =  ['M','N']
place_images['mumbai']=['P','Q']

category_label = {0:'Open Space',1:'Non-Residential',                   2:'Residential Atomistic',3:'Residential Informal Subdivision',                   4:'Residential Formal Subdivision',5:'Residential Housing Project',                   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'}

cats_map = {}
cats_map[0] = 0
cats_map[1] = 1
cats_map[2] = 2
cats_map[3] = 2
cats_map[4] = 2
cats_map[5] = 3

place_images = {}
place_images['hindupur']=['W', 'X']#['W', 'X', 'Y', 'Z']
place_images['singrauli']=['Q','R']#['Q','R','S','T','U']
place_images['jaipur']=['W','X']#['W','X','Y','Z']
place_images['hyderabad']=['R','S']#['R','S','T','U']
place_images['sitapur']=['T','U']#['T','U','V']
place_images['kanpur']=['AL', 'AM']#['AL', 'AM', 'AN']
place_images['belgaum']=['R','S']#['R','S','T']
place_images['parbhani']=['W','X']#['W','X','Y','Z']
place_images['pune']=['U', 'S']
place_images['ahmedabad']= ['W', 'X']#['W', 'X', 'Y', 'AA']
place_images['malegaon']=  ['X', 'Y']#['X', 'Y', 'Z']
place_images['kolkata'] =  ['O','P']#['O','P','Q','R']
place_images['mumbai']=['R','S']#['R','S','U','V']

# In[11]:


df_t=pd.read_csv(TRAIN_DS)
df_v=pd.read_csv(VALID_DS)
print(TRAIN_DS,df_t.shape)
print(VALID_DS,df_v.shape)

# In[12]:


train_gen = batch_generator.BatchGenerator(
    df_t,
    remapping=REMAPPING,
    look_window=window,
    batch_size=batch_size,
    one_hot=3,
    bands_last=BANDS_LAST)
valid_gen = batch_generator.BatchGenerator(
    df_v,
    remapping=REMAPPING,
    look_window=window,
    batch_size=batch_size,
    one_hot=3,
    bands_last=BANDS_LAST)
inpts,targs=next(iter(train_gen))


# In[13]:


print("train-shape,steps:",train_gen.dataframe.shape,train_gen.steps)



# In[14]:


network=util_network.build_xmodel(
    input_shape=INPUT_SHAPE,
    output_nodes=NB_CATS,
    input_conv_block=True,
    print_summary=False)


# In[15]:


category_weights =  util_training.generate_category_weights(df_t,remapping='standard',log=False,mu=1.0,max_score=None)
weights = list(zip(*category_weights.items()))[1]
loss = util_training.make_loss_function_wcc(weights)
print(weights,loss)
util_network.compile_network(network, loss, LR=0.001)



# In[16]:


weights_label='dev_weights_fast'
epochs=10


# In[17]:


callbacks, weights_path = util_training.create_callbacks(
    data_root, 
    model_id, 
    weights_label=weights_label, 
    patience=2)


# In[18]:

section('FAST TRAIN')

history_fast = network.fit_generator(
    train_gen, 
    epochs=epochs, 
    callbacks=callbacks, 
    steps_per_epoch=train_gen.steps,
    validation_data=valid_gen, 
    validation_steps=valid_gen.steps,    
    shuffle=True,
    use_multiprocessing=True,
    max_queue_size=40,
    workers=64 )


# In[19]:

fast_name=model_name(TRAIN_DS,'fast')
network.save(fast_name, include_optimizer=True)
print("MODEL SAVED: ",fast_name)

# In[21]:


network=util_network.build_xmodel(
    input_shape=INPUT_SHAPE,
    output_nodes=NB_CATS,
    input_conv_block=True,
    print_summary=False)
network.load_weights(weights_path)
util_network.compile_network(network, loss, LR=0.0001)


# In[22]:


train_gen.reset()
valid_gen.reset()


# In[23]:


weights_label='weights_slow'
callbacks, weights_path = util_training.create_callbacks(
    data_root, 
    model_id, 
    weights_label=weights_label, 
    patience=5)


# In[24]:


section('SLOW TRAIN')
history_slow = network.fit_generator(
    train_gen, 
    epochs=epochs, 
    callbacks=callbacks, 
    steps_per_epoch=train_gen.steps,
    validation_data=valid_gen, 
    validation_steps=valid_gen.steps,    
    shuffle=True,
    use_multiprocessing=True,
    max_queue_size=40,
    workers=64 )


# In[25]:

slow_name=model_name(TRAIN_DS,'slow')
network.save(slow_name, include_optimizer=True)
print("MODEL SAVED: ",slow_name)


# In[27]:

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





