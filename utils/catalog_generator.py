import os, sys
import pandas as pd
import numpy as np
import math
import rasterio as rio
import gdal
from tensorflow.keras.utils import to_categorical
import threading
import tensorflow.keras as keras
import utils.util_rasters as util_rasters
import utils.util_imagery as util_imagery

# class itself
class CatalogGenerator(keras.utils.Sequence):
    
    # constructor stuff
    def __init__(self,
                df,
                batch_size=128,
                look_window=17,
                remapping=None,
                one_hot=4,
                flatten=False,
                bands_first=False
                ):
        self.batch_size=batch_size
        self.look_window=look_window
        self.look_radius=int(look_window//2)
        
        if remapping is None:
            self.remapping = remapping
        else:
            if isinstance(remapping, dict):
                self.remapping = remapping
            elif isinstance(remapping, str):
                remapping_lower = remapping.lower()
                if remapping_lower in ['standard','residential','3cat','3category']:
                    self.remapping = {0:0,1:1,2:2,3:2,4:2,5:2,6:6}
                elif remapping_lower == 'roads':
                    self.remapping = {0:0,1:0,2:0,3:0,4:0,5:0,6:1}
                else:
                    raise ValueError('Unrecognized remapping identifier: ',remapping)
            else:
                raise ValueError('Illegal object passed as remapping: ',remapping)
        assert isinstance(one_hot, int)
        self.one_hot = one_hot
        self.flatten = flatten
        self.bands_first=bands_first
        self._set_data(df)
        
    def _set_data(self,df):
        if isinstance(df,str):
            self.dataframe=pd.read_csv(data)
        else:
            self.dataframe=df
        self.size=len(self.dataframe.index)
        self.steps=int(np.ceil(self.size/self.batch_size))
        self.reset()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # this may need to be increased by one
        return self.steps


    def reset(self):
        self.rows=None
        self.batch_index=-1
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
            #print('last batch: start '+str(start)+', end '+str(end))
        self.rows=self.dataframe.iloc[start:end]
        inputs=self._get_inputs()
        targets=self._get_targets()
        return inputs, targets

    def _get_inputs(self):
        imgs=[]
        for path in self.rows.path:
            im = self._read_image(path)
            im = self._construct_sample(im)
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
        if not self.bands_first:
            im=im.swapaxes(0,1).swapaxes(1,2)
        return im
    
    # simple example of more customized input generator
    def _construct_sample(self, image):

        if not self.bands_first:
            assert image.shape[0] == image.shape[1]
        else:
            assert image.shape[1] == image.shape[2]

        image = util_imagery.s2_preprocess(image, bands_first=self.bands_first)

        # grab look window
        image_side = image.shape[1]
        center = int(image_side//2)
        return util_rasters.window(
            image,
            center,
            center,
            self.look_radius,
            bands_first=self.bands_first)

    def _get_targets(self):
        categories = list(self.rows.lulc)
        targets = np.array(categories)
        if self.remapping is not None:
            for k in sorted(self.remapping.keys()):
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
    
