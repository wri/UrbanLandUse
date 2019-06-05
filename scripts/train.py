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
#
# CONSTANTS
#
REMAPPING='3cat'
NB_CATS=3
BANDS_LAST=True
EPOCHS=10
PATIENCE=2
DATASETS_DIR=f'{PROJECT_DIR}/datasets'
TRAIN_DS=f'{DATASETS_DIR}/multi_city_2img-train.csv'
VALID_DS=f'{DATASETS_DIR}/multi_city_2img-valid.csv'
DEV_TRAIN_DS=f'{DATASETS_DIR}/multi_city_2img-8-train.csv'
DEV_VALID_DS=f'{DATASETS_DIR}/multi_city_2img-2-valid.csv'
FAST_WEIGHTS_LABEL='weights_fast'
SLOW_WEIGHTS_LABEL='weights_slow'
FAST_LR=1e-3
SLOW_LR=1e-4
SCORECARD_PATH='/data/phase_iv/models/scorecard_phase_iv_models.csv'
TS_FMT="%Y%m%d-%H:%M:%S"
if BANDS_LAST:
    INPUT_SHAPE=(17,17,6)
    K.set_image_data_format('channels_last')
else:
    INPUT_SHAPE=(6,17,17)
    K.set_image_data_format('channels_first')



#
# METHODS
#
def get_config(cfig,*keys):
    """ load config file
    """
    config=yaml.safe_load(open('configs/{}.yaml'.format(cfig)))
    for key in keys:
        config=config[key]
    return config


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


def generators_category_weights(train_ds,valid_ds,window,batch_size):
    df_t=pd.read_csv(train_ds)
    df_v=pd.read_csv(valid_ds)
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
    category_weights=util_training.generate_category_weights(
        df_t,
        remapping='standard',
        log=False,
        mu=1.0,
        max_score=None)
    category_weights=list(zip(*category_weights.items()))[1]
    section('generators_category_weights')
    print(train_ds,df_t.shape)
    print(valid_ds,df_v.shape)
    print("train-shape,steps:",train_gen.dataframe.shape,train_gen.steps)
    print('category_weights:',category_weights)
    return train_gen, valid_gen, category_weights


def build_network(category_weights,lr,weights_path=None):
    network=util_network.build_xmodel(
        input_shape=INPUT_SHAPE,
        output_nodes=NB_CATS,
        input_conv_block=True,
        print_summary=False)
    if weights_path:
        network.load_weights(weights_path)
    loss=util_training.make_loss_function_wcc(category_weights)
    util_network.compile_network(network, loss, LR=lr)
    return network


def train_network(
        train_gen,
        valid_gen,
        data_root,
        model_id,
        train_ds,
        weights_label,
        batch_size,
        callbacks):
    callbacks, weights_path = util_training.create_callbacks(
        data_root=data_root, 
        model_id=model_id, 
        weights_label=weights_label, 
        patience=PATIENCE)
    section(f'TRAIN: {weights_label}')
    history_fast = network.fit_generator(
        train_gen, 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        steps_per_epoch=train_gen.steps,
        validation_data=valid_gen, 
        validation_steps=valid_gen.steps,    
        shuffle=True,
        use_multiprocessing=True,
        max_queue_size=40,
        workers=64 )
    model_name=model_name(train_ds,'fast')
    network.save(model_name, include_optimizer=True)
    print("MODEL SAVED: ",model_name)


def run(cfig,dev=True,fast=True):
    #
    # setup
    #
    section('SETUP')
    print(PROJECT_DIR)
    print(tf.__version__)
    print("BANDS_LAST:",BANDS_LAST,"DATA-FORMAT:",K.image_data_format())
    cfig=get_config(cfig)
    meta=cfig['meta']
    setup=cfig['setup']
    places=cfig['places']
    model_id=meta['model_id']
    data_root=meta['data_root']
    window=setup['window']
    batch_size=setup['batch_size']
    if fast:
        lr=FAST_LR
        weights_label=FAST_WEIGHTS_LABEL='weights_fast'
        weights_path=None
    else:
        lr=SLOW_LR
        weights_label=SLOW_WEIGHTS_LABEL
        _,weights_path=util_training.create_callbacks(
            data_root=data_root, 
            model_id=model_id, 
            weights_label=FAST_WEIGHTS_LABEL, 
            patience=2)
        del(_)
    if dev:
        train_ds=DEV_TRAIN_DS
        valid_ds=DEV_VALID_DS
        weights_label=f'dev_{weights_label}'
    else:
        train_ds=TRAIN_DS
        valid_ds=VALID_DS
    #
    # data/network/train
    #
    train_gen, valid_gen, category_weights=generators_category_weights(
            train_ds=train_ds,
            valid_ds=valid_ds,
            window=window,
            batch_size=batch_size
        )
    network=build_network(
        category_weights=category_weights,
        lr=lr,
        weights_path=weights_path)
    train_network(
        data_root=data_root,
        model_id=model_id,
        train_ds=train_ds,
        weights_label=weights_label)


