import os, sys
import pandas as pd
import numpy as np
import math
import rasterio as rio
import gdal
from tensorflow.keras.utils import to_categorical
import threading
import tensorflow.keras

ULU_REPO = os.environ["ULU_REPO"]
sys.path.append(ULU_REPO+'/utils')
import util_rasters

# class itself
class ImageSampleGenerator(keras.utils.Sequence):
    
    # constructor stuff
    def __init__(self,
                image,
                pad=32,

                look_window=17,
                prep_image=False,
                ):
        if prep_image:
            self.image = self._prep_image(image)
        else:
            self.image = image
        self.pad=pad
        self.look_window=look_window
        self._set_data(image)
    
    # eventually this should all be happening beforehand
    # want to just pass the prepared, fused input_stack to generator constructor
    def _prep_image(self,
                    image):
        # drop alpha
        image = image[:-1,:,:]
        # manual "rescaling" as in all previous phases
        image = image.astype('float32')
        image = image/10000.
        image = np.clip(image,0.0,1.0)
        #print 'image prepped'
        return image

    def _set_data(self,
                  image,):
        assert isinstance(image,np.ndarray)
        # can relax conditions later
        assert len(image.shape)==3
        assert image.shape[1]==image.shape[2]
        self.batch_size=(image.shape[1]-self.pad-self.pad)
        self.size = self.batch_size^2
        # for starters, will make columns into batches/steps
        self.steps= self.batch_size
        #print 'data set'
        self.reset()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # this may need to be increased by one
        return self.steps

    def reset(self):
        self.batch_index=-1

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= self.steps or index < 0:
            raise ValueError('illegal batch index:',str(index))
        self.batch_index = index
        inputs=self._get_inputs(index)
        #print 'inputs', inputs.shape
        #print 'batch #'+str(index)+' generated with shape '+str(inputs.shape)
        return inputs

    def _get_inputs(self, index):
        look_radius = self.look_window/2
        samples=[]
        for j in range(self.pad,self.image.shape[1]-self.pad):
            sample = util_rasters.window(self.image,j,index+self.pad,look_radius,bands_first=True)
            samples.append(sample)
        return np.array(samples)

    
