import os, sys
import pandas as pd
import numpy as np
import math
import rasterio as rio
import gdal
from tensorflow.keras.utils import to_categorical
import threading
import tensorflow.keras as keras

ULU_REPO = os.environ["ULU_REPO"]
sys.path.append(ULU_REPO+'/utils')
import util_rasters

# class itself
class TilewiseGenerator(keras.utils.Sequence):
    
    # constructor stuff
    def __init__(self,
                place,
                image_suffix,
                tiles,

                data_root='/data/phase_iv/',
                processing='none',
                source='s2',

                look_window=17,
                ):
        # if prep_image:
        #     self.image = self._prep_image(image)
        # else:
        #     self.image = image
        self.place = place
        self.image_suffix = image_suffix
        self.tiles = tiles

        self.tile_properties = tiles['features'][0]['properties']
        self.tile_resolution = int(self.tile_properties['resolution'])
        self.tile_side = int(self.tile_properties['tilesize'])
        self.tile_pad = int(self.tile_properties['pad'])
        self.batch_size = self.tile_side * self.tile_side
        if self.tile_resolution == 10:
            self.zfill = 3
        elif self.tile_resolution == 5:
            self.zfill = 4
        elif self.tile_resolution == 1:
            self.zfill = 5
        else:
            raise ValueError('tiles have unexpected resolution: '+self.tile_resolution)
        
        self.data_root = data_root
        self.processing = str(processing).lower()
        self.source = source

        self.look_window=look_window

        self.look_radius = look_window / 2

        # simply the number of tiles
        self.steps = len(tiles['features'])

        self.reset()

        # self._set_data(image)


    def _set_data(self,
                  image,):
        self.batch_size=(image.shape[1]-self.pad-self.pad)
        self.size = self.batch_size^2
        # for starters, will make columns into batches/steps
        self.steps= self.batch_size
        #print('data set')
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
        #print('inputs', inputs.shape)
        #print('batch #'+str(index)+' generated with shape '+str(inputs.shape))
        return inputs

    def _get_inputs(self, index):
        image_tile_path = self._get_input_path(index)
        image, geo, prj, cols, rows = util_rasters.load_geotiff(image_tile_path,dtype='uint16')
        image = self._prep_image(image)

        assert isinstance(image,np.ndarray)
        # can relax conditions later
        assert len(image.shape)==3
        assert image.shape[0]==6
        assert image.shape[1]==image.shape[2]
        assert image.shape[1]==self.tile_side + (2 * self.tile_pad)

        samples=[]
        for j in range(self.tile_pad,image.shape[1]-self.tile_pad):
            #for i in range(self.tile_pad,self.tile_pad+1):
            for i in range(self.tile_pad,image.shape[1]-self.tile_pad):
                sample = util_rasters.window(image,j,i,self.look_radius,bands_first=True)
                samples.append(sample)
        return np.array(samples)


    def _get_input_path(self, index):
        image_tile_path = self.data_root+self.place+'/imagery/'+self.processing+'/'+\
            self.place+'_'+self.source+'_'+self.image_suffix+'_'+str(self.tile_resolution)+'m'+'_'+'p'+str(self.tile_pad)+'_'+\
            'tile'+str(index).zfill(self.zfill)+'.tif'
        #print(image_tile_path)
        return image_tile_path

    def _prep_image(self,
                    image):
        # drop alpha
        image = image[:-1,:,:]
        # manual "rescaling" as in all previous phases
        image = image.astype('float32')
        image = image/10000.
        image = np.clip(image,0.0,1.0)
        #print('image prepped')
        return image

    
