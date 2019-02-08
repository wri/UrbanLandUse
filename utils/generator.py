import os, sys
import pandas as pd
import numpy as np
import math
import rasterio as rio
import gdal
from keras.utils import to_categorical
import threading


ULU_REPO = os.environ["ULU_REPO"]
sys.path.append(ULU_REPO+'/utils')
import util_rasters

# class itself
class SampleGenerator(object):
    
    # constructor stuff
    def __init__(self,
                df,
                batch_size=128,
                look_window=17,
                remapping=None,
                one_hot=True,
                lock=None
                ):
        if lock is None:
            self.lock = threading.Lock()
        else:
            self.lock=lock

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
        self.one_hot = one_hot
        self._set_data(df)
        
    def _set_data(self,
                  df):
        """ set data
            * read csv
            * set input/target_paths
            * reset(shuffle,etc...)
        """
        if isinstance(df,str):
            self.dataframe=pd.read_csv(data)
        else:
            self.dataframe=df
        self.size=self.dataframe.shape[0]
        self.steps=int(math.floor(1.0*self.size/self.batch_size))+1
        self.reset()

    def reset(self):
        """ reset the generator
            * reset batch index to -1
            * shuffle input/target paths
        """
        self.profiles=None
        self.rows=None
        self.batch_index=-1
        # randomization step (?)
        self.dataframe=self.dataframe.sample(frac=1)
        
    # iteration stuff
    def __next__(self):
        """ batchwise return tuple of (inputs,targets)
        """        
        with self.lock:
            start,end=self._batch_range()
        self.rows=self.dataframe.iloc[start:end]
        inputs,self.profiles=self._get_inputs()
        targets=self._get_targets()
        #print 'inputs', inputs.shape
        #print 'targets', targets.shape
        return inputs, targets
        
    def _batch_range(self):
        """ batch setup
            * reset if batch_index >= # of steps
            * increment batch_index
            * start/end indices for batch_index
        """
        if (self.batch_index+1>=self.steps):
            print 'resetting batch stuff from next'
            self.reset()
        self.batch_index+=1
        print 'batch index', self.batch_index
        start=self.batch_index*self.batch_size
        end=start+self.batch_size
        if end >= self.size:
            end = self.size-1
            print 'last batch: start '+str(start)+', end '+str(end)
        if (self.batch_index+1>=self.steps):
            end=self.size-1
            print 'last batch: start '+str(start)+', end '+str(end)
        else:
            end=start+self.batch_size
        return start,end

    def _get_inputs(self):
        imgs=[]
        profiles=[]
        for path in self.rows.path:
            im=self._read_image(path)
            im2 = self._construct_sample(im, self.look_window/2)
            imgs.append(im2)
            #profiles.append(profile)
        return np.array(imgs),None
    
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

    def _read_image_old(self,path):
        with rio.open(path,'r') as src:
            profile=src.profile
            im=src.read()
        #bands last
        im = image.swapaxes(0,1).swapaxes(1,2)
        return im, profile
    
    # simple example of more customized input generator
    def _construct_sample(self, image, look_radius):
        assert image.shape[1] == image.shape[2]

        # manual "rescaling" as in all previous phases
        image = image.astype('float32')
        image = image/10000.
        image = np.clip(image,0.0,1.0)

        # remove alpha
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

    # other
    def __iter__(self):
        """py2: define as iterator"""
        return self
    
    def next(self):
        """py2: alias __next__"""
        return self.__next__()

    
    # subsequently added
    def get_label_series(self):
        mapped_series = self.dataframe['lulc'].map(self.remapping)
        if mapped_series.isnull().sum() > 0:
            raise KeyError('remapping does not cover all relevant values, resulting in nan entries')
        return mapped_series
    
