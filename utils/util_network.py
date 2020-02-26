# topic: keras-specific components

import warnings
warnings.filterwarnings('ignore')

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
#import tensorflow.tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.layers import Input, Add, Lambda

import tensorflow as tf


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

# xception block stuff
def xception_block(x,filters,depth=2,input_act=True,dropout=False):
    residual=_pointwise(x,filters)
    x=_depthwise(x,filters,depth,input_act)
    x=MaxPooling2D(3,strides=2,padding='same')(x)
    x=keras.layers.add([x, residual])
    if dropout:
        x=Dropout(0.25)(x)
    return x

def _depthwise(x,filters,depth,input_act):
    for i in range(depth):
        if input_act or i:
            x = Activation('relu')(x)
        x = SeparableConv2D(filters,3,padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
    return x 

def _pointwise(x,filters):
    x = Conv2D(filters,1,strides=2,padding='same',use_bias=False)(x)
    return BatchNormalization()(x)

def build_model(cblock,filters1=32,filters2=64,print_summary=True,input_shape=(6,17,17),output_nodes=3):
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

# alternate structure with xception block
def build_xmodel(
        filters1=32,
        filters2=64,
        print_summary=True,
        input_shape=(6,17,17),
        output_nodes=3,
        input_conv_block=False,
        input_conv_filters=32):
    inputs=Input(shape=input_shape)
    x=inputs
    if input_conv_block:
        x=conv_block(input_conv_filters,x)
    x=xception_block(x,filters1,input_act=False)
    x=xception_block(x,filters2)
    x=keras.layers.Flatten()(x)
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