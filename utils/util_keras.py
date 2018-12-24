# topic: keras-specific components

import warnings
warnings.filterwarnings('ignore')

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Add, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, History

import tensorflow as tf


def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy
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



def pool_dropout(x):
    x=MaxPooling2D(pool_size=(2, 2))(x)
    return Dropout(0.25)(x)

def denselayers(x,output_nodes=4):
    x=Dense(512)(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(output_nodes)(x)
    return x

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


def build_model(cblock,filters1=32,filters2=64,print_summary=True,input_shape=(8,17,17),output_nodes=4):
    inputs=Input(shape=input_shape) 
    x=cblock(filters1,inputs)
    x=pool_dropout(x)
    x=cblock(filters2,x)
    x=pool_dropout(x)
    x=Flatten()(x)
    x=denselayers(x,output_nodes)
    if output_nodes == 1:

        x = Activation('sigmoid')(x)
    else:
        x = Activation('softmax')(x)

    m=Model(inputs=inputs, outputs=x)
    if print_summary:
        m.summary()
    return m

def compile_network(network, loss, LR=0.001, metrics=['accuracy']):

    opt = keras.optimizers.Adam(lr=LR)
    network.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)


def create_callbacks(data_root, model_id, weights_label='WCC_weights.best', patience=4):
    filepath = data_root+'models/'+model_id+'_'+weights_label+'.hdf5'
    estop_cb=EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    save_best_cb=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, 
                                 mode='auto', period=1)
    history_cb=History()
    return [estop_cb,save_best_cb,history_cb], filepath
