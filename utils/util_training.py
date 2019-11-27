import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
import tensorflow as tf

# loss function stuff

# get category counts from chip catalog
def calc_category_counts(df,remapping='standard'):
    count_series = df['lulc'].value_counts()
    counts_dict = count_series.to_dict()
    if remapping is None:
        return counts_dict
    else:
        if isinstance(remapping, str):
            remapping_lower = remapping.lower()
            if remapping_lower in ['standard','residential','3cat','3category']:
                remapping = {0:0,1:1,2:2,3:2,4:2,5:2,6:6}
            elif remapping_lower == 'roads':
                remapping = {0:0,1:0,2:0,3:0,4:0,5:0,6:1}
            else:
                raise ValueError('Unrecognized remapping identifier: ',remapping)
        assert isinstance(remapping, dict)
        recount_dict = {}
        for k in sorted(counts_dict.keys()):
            if remapping[k] not in recount_dict:
                recount_dict[remapping[k]] = 0
            recount_dict[remapping[k]] += counts_dict[k]
        return recount_dict

# calculate weights from category counts
def calc_category_weights(category_counts,log=False,mu=1.0):
    assert isinstance(category_counts,dict)
    n_samples = sum(category_counts.values())
    category_weights = {}
    for k in sorted(category_counts.keys()):
        # for the moment skipping scenario where one or more cats have zero samples
        count = category_counts[k]
        score = n_samples / float(count)
        if log:
            score = math.log(mu*score)
        category_weights[k] = score
    return category_weights

# normalize provided category weights
def normalize_category_weights(category_weights,max_score=None):
    assert isinstance(category_weights,dict)
    min_weight = min(category_weights.values())
    normalized_weights = {}
    for k in sorted(category_weights.keys()):
        normalized_weights[k] = category_weights[k]/min_weight
        if max_score is not None:
            normalized_weights[k] = min(max_score, normalized_weights[k])
    return normalized_weights

# single function to wrap previous three: catalog -> normalized weights
def generate_category_weights(df,remapping='standard',log=False,mu=1.0,max_score=None):
    cat_counts = calc_category_counts(df,remapping=remapping)
    cat_weights = calc_category_weights(cat_counts,log=log,mu=1.0)
    cat_normed = normalize_category_weights(cat_weights,max_score=max_score)
    return cat_normed

# build loss function
def make_loss_function_wcc(weights):
    """ make loss function: weighted categorical crossentropy
        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if isinstance(weights,list) or isinstance(weights,np.ndarray):
        weights=K.variable(weights)

    def loss(target,output,from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')

    return loss

# returns callback functions for training, as well as path of weights file
def create_callbacks(data_root, model_id, weights_label='weights', patience=4):
    filepath = data_root+'models/'+model_id+'_'+weights_label+'.hdf5'
    estop_cb=EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    save_best_cb=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, 
                                 mode='auto', period=1)
    history_cb=History()
    return [estop_cb,save_best_cb,history_cb], filepath
