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


def build_model(cblock,filters1=32,filters2=64,print_summary=True,input_shape=(8,17,17),output_nodes=4):
    inputs=Input(shape=input_shape) 
    x=cblock(filters1,inputs)
    x=pool_dropout(x)
    x=cblock(filters2,x)
    x=pool_dropout(x)
    x=Flatten()(x)
    x=denselayers(x,output_nodes)
    m=Model(inputs=inputs, outputs=x)
    if print_summary:
        m.summary()
    return m

