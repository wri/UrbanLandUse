from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import numpy as np
from tensorflow.python.keras.utils import Sequence

import utils.util_imagery as util_imagery
import utils.util_rasters as util_rasters
# CONSTANTS
#
WINDOW_PADDING='window'


#
# Helpers
#

def get_padding(pad,window):
    if pad==WINDOW_PADDING:
        return int(window/2)
    else:
        return pad

#
# ImageGenerator
#
class ImageGenerator(Sequence):
    
    # constructor stuff
    def __init__(self,
                image,
                pad=WINDOW_PADDING,
                look_window=17,
                bands_first=False,
                preprocess=util_imagery.s2_preprocess):
        assert image.ndim==3
        self.preprocess=preprocess
        if preprocess is not None:
            image=self.preprocess(image,bands_first=bands_first)
        self.image=image
        self.pad=get_padding(pad,look_window)
        self.look_window=look_window
        self.look_radius=int(look_window/2)
        self.bands_first=bands_first
        self._set_data(image)


    def _set_data(self,image):
        assert isinstance(image,np.ndarray)
        # can relax conditions later
        if self.bands_first:
            assert image.shape[1]==image.shape[2]
        else:
            assert image.shape[0]==image.shape[1]
        self.batch_size=(image.shape[1]-self.pad-self.pad)
        self.size=self.batch_size^2
        # for starters, will make columns into batches/steps
        self.steps=self.batch_size
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
        self.batch_index=index
        inputs=self._get_inputs(index)
        return inputs


    def _get_inputs(self, index):
        samples=[]
        for j in range(self.pad,self.image.shape[1]-self.pad):
            sample=util_rasters.window(
                self.image,
                j,index+self.pad,
                self.look_radius,
                bands_first=(self.bands_first))
            samples.append(sample)
        return np.array(samples)
