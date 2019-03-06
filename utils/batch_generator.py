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
class BatchGenerator(keras.utils.Sequence):
    
    # constructor stuff
    def __init__(self,
                df,
                batch_size=128,
                look_window=17,
                remapping=None,
                one_hot=4,
                flatten=False
                ):

        self.batch_size=batch_size
        self.look_window=look_window
        
        if remapping is None:
            self.remapping = remapping
        else:
            if isinstance(remapping, dict):
                self.remapping = remapping
            elif isinstance(remapping, str):
                if remapping.lower() == '3cat' or remapping.lower() == '3category':
                    self.remapping = {0:0,1:1,2:2,3:2,4:2,5:2,6:3}
                elif remapping.lower() == 'roads':
                    self.remapping = {0:0,1:0,2:0,3:0,4:0,5:0,6:1}
                else:
                    raise ValueError('Unrecognized remapping identifier: ',remapping)
            else:
                raise ValueError('Illegal object passed as remapping: ',remapping)
        assert isinstance(one_hot, int)
        self.one_hot = one_hot
        self.flatten = flatten
        self._set_data(df)
        
    def _set_data(self,
                  df):
        if isinstance(df,str):
            self.dataframe=pd.read_csv(data)
        else:
            self.dataframe=df
        self.size=len(self.dataframe.index)
        self.steps=int(np.floor(self.size/self.batch_size))+1
        self.reset()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # this may need to be increased by one
        return self.steps

    def reset(self):
        self.rows=None
        self.batch_index=-1
        # randomization step (?)
        self.dataframe=self.dataframe.sample(frac=1)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index >= self.steps or index < 0:
            raise ValueError('illegal batch index:',str(index))
        self.batch_index = index
        start=self.batch_index*self.batch_size
        end=start+self.batch_size
        if end >= self.size:
            # remember this interval is used [,) so want size as final index (not size-1)
            end = self.size
            #print 'last batch: start '+str(start)+', end '+str(end)

        self.rows=self.dataframe.iloc[start:end]
        inputs=self._get_inputs()
        targets=self._get_targets()
        #print 'inputs', inputs.shape
        #print 'targets', targets.shape

        return inputs, targets

    def _get_inputs(self):
        imgs=[]
        for path in self.rows.path:
            im = self._read_image(path)
            im = self._construct_sample(im, self.look_window/2)
            if self.flatten:
                imgs.append(im.flatten())
            else:
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
        if self.one_hot != 0:
            # keras.utils.to_categorical(y, num_classes=None, dtype='float32')
            targets = to_categorical(targets, num_classes=self.one_hot)
        return targets

        
    
    # other functions

    def get_label_series(self):
        if self.remapping is not None:
            mapped_series = self.dataframe['lulc'].map(self.remapping)
        else:
            mapped_series = self.dataframe['lulc']
        if mapped_series.isnull().sum() > 0:
            raise KeyError('remapping does not cover all relevant values, resulting in nan entries')
        return mapped_series
    
