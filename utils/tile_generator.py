import os, sys
import pandas as pd
import numpy as np
import math
import rasterio as rio
import gdal
from keras.utils import to_categorical
import threading
import keras

ULU_REPO = os.environ["ULU_REPO"]
sys.path.append(ULU_REPO+'/utils')
import util_rasters

# class itself
class TileGenerator(keras.utils.Sequence):
    
    # constructor stuff
    def __init__(self,
                input_stack,
                pad=32,

                batch_size=128,
                look_window=17,
                ):
        self.pad = pad
        self.look_window=look_window
        
        self._set_data(input_stack,pad)
        
    def _set_data(self,
                  input_stack,
                  pad):
        assert isinstance(input_stack,np.array)
        # can relax conditions later
        assert len(input_stack.shape==3)
        assert input_stack.shape[1]=input_stack.shape[2]
        self.image = input_stack[:,pad:-pad,pad:-pad]
        self.size = image.shape[1]^2
        self.batch_size=image.shape[1]
        # for starters, will make columns into batches/steps
        self.steps=int(image.shape[2])
        self.reset()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # this may need to be increased by one
        return self.steps

    def reset(self):
        self.batch_index=-1
        # randomization step (?)
        self.dataframe=self.dataframe.sample(frac=1)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= self.steps or index < 0:
            raise ValueError('illegal batch index:',str(index))
        self.batch_index = index
        inputs=self._get_inputs()
        #print 'inputs', inputs.shape
        return inputs, targets

    def _get_inputs(self):
        imgs=[]
        for path in self.rows.path:
            im = self._read_image(path)
            im = self._construct_sample(im, self.look_window/2)
            imgs.append(im)
        return np.array(imgs)
    
    def _read_image(self,path,dtype='uint16'):
        # read using gdal directly
        # skip loading any nonessential elements
        # can add this back in later if we use metadata
        obj = gdal.Open(path, gdal.gdalconst.GA_ReadOnly)
        #prj = obj.GetProjection()
        #geotrans = obj.GetGeoTransform()
        #cols = obj.RasterXSize
        #rows = obj.RasterYSize
        #im = np.zeros((rows,cols), dtype=dtype)
        im = obj.ReadAsArray().astype(dtype)
        return im
    
    # simple example of more customized input generator
    def _construct_sample(self, image, look_radius):
        assert image.shape[1] == image.shape[2]

        # manual "rescaling" as in all previous phases
        image = image.astype('float32')
        image = image/10000.
        image = np.clip(image,0.0,1.0)

        # drop alpha
        image = image[:-1,:,:]

        # grab look window
        image_side = image.shape[1]
        center = image_side/2
        return util_rasters.window(image,center,center,look_radius,bands_first=True)

    def _get_targets(self):
        categories = list(self.rows.lulc)
        targets = np.array(categories)
        if self.remapping is not None:
            for k in sorted(self.remapping.iterkeys()):
                targets[targets==k] = self.remapping[k]
        if self.one_hot:
            targets = to_categorical(targets)
        return targets

        
    
    # other functions

    def get_label_series(self):
        mapped_series = self.dataframe['lulc'].map(self.remapping)
        if mapped_series.isnull().sum() > 0:
            raise KeyError('remapping does not cover all relevant values, resulting in nan entries')
        return mapped_series
    
