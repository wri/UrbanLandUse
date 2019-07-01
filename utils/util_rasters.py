# organized set of helper functions
# drawn from bronco.py and bronco notebooks
# topic: rasters

import warnings
warnings.filterwarnings('ignore')
#
#import os
import sys
#import json
#import itertools
#import pickle
#from pprint import pprint
#
import numpy as np
#import shapely
#import cartopy
from osgeo import gdal
import matplotlib.pyplot as plt
#
import descarteslabs as dl

from urllib3.exceptions import ProtocolError

import subprocess

# FILE READ/WRITE

def load_geotiff(tif,dtype='uint16'):
    obj = gdal.Open(tif, gdal.gdalconst.GA_ReadOnly)
    prj = obj.GetProjection()
    geo = obj.GetGeoTransform()
    cols = obj.RasterXSize
    rows = obj.RasterYSize
    img = np.zeros((rows,cols), dtype=dtype)
    img = obj.ReadAsArray().astype(dtype)
    del obj
    return img, geo, prj, cols, rows

def write_1band_geotiff(outfile, img, geo, prj, data_type=gdal.GDT_Byte):
    driver = gdal.GetDriverByName('GTiff')
    rows = img.shape[0]
    cols = img.shape[1]
    outds = [ ]
    if (data_type == gdal.GDT_Byte):
        opts = ["INTERLEAVE=BAND", "COMPRESS=LZW", "PREDICTOR=1", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        outds  = driver.Create(outfile, cols, rows, 1, data_type, options=opts)
    else:
        outds  = driver.Create(outfile, cols, rows, 1, data_type)
    outds.SetGeoTransform(geo)
    outds.SetProjection(prj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(img)
    del outds

def write_multiband_geotiff(outfile, img, geotrans, prj, data_type=gdal.GDT_Byte):
    driver = gdal.GetDriverByName('GTiff')
    bands = img.shape[0]
    rows = img.shape[1]
    cols = img.shape[2]
    outds = [ ]
    if (data_type == gdal.GDT_Byte):
        opts = ["INTERLEAVE=BAND", "COMPRESS=LZW", "PREDICTOR=1", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        outds  = driver.Create(outfile, cols, rows, bands, data_type, options=opts)
    else:
        #print(outfile, cols, rows, bands, data_type)
        outds  = driver.Create(outfile, cols, rows, bands, data_type)
    outds.SetGeoTransform(geotrans)
    outds.SetProjection(prj)
    for b in range(bands):
        outds.GetRasterBand(b+1).WriteArray(img[b])
    del outds

# assumes first dimension is bands
def window(x,j,i,r,bands_first=True):
    ystart,yend=int(j-r),int(j+r+1)
    xstart,xend=int(i-r),int(i+r+1)
    if bands_first:
        w = x[:,ystart:yend,xstart:xend]
    else:
        w = x[ystart:yend,xstart:xend,:]
    return w

def maxmin_info(img):
    mns=np.min(img,axis=(0,1))
    mxs=np.max(img,axis=(0,1))
    print('band, min, max')
    for i,(mn,mx) in enumerate(zip(mns,mxs)):
        print(i,mn,mx)

def stats_byte_raster(y, 
        category_label={0:'Open Space',1:'Non-Residential',\
                   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
                   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
                   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'},
        band_index=0):
    if isinstance(y,str):
            y,_,_,_,_=load_geotiff(y,dtype='uint8')
    if y.dtype != 'uint8':
        raise ArgumentException('passed raster is not byte-valued:'+str(y.dtype))
    if y.ndim == 3:
        y = y[band_index]
    yd = {}
    for c in range(256):
        if np.sum((y == c))>0:
            yd[c] = np.sum((y == c))
            print(c, yd[c], category_label[c] if c in category_label else '')
    return yd

def stats_byte_tiles(data_path, place, tiles, label_suffix, 
        no_data=255,
        categories=[0,1,2,3,4,5,6],
        category_label={0:'Open Space',1:'Non-Residential',\
                   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
                   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
                   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'}):

    tile_size = tiles['features'][0]['properties']['tilesize']
    tile_pad = tiles['features'][0]['properties']['pad']
    tile_pixels = (tile_size + tile_pad*2)**2

    tile_stats = {}
    for tile_id in range(len(tiles['features'])):
        tile = tiles['features'][tile_id]
        label_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_'+label_suffix+'.tif'
        tile_stats[tile_id] = stats_byte_raster(label_file, category_label, show=False)

    categories = [0,1,2,3,4,5,6]

    agg_stats = {}
    for c in categories:
        agg_stats[c] = 0

    for tile_id in tile_stats.keys():
        if tile_stats[tile_id][no_data] != tile_pixels:
            print(tile_id, tile_stats[tile_id])
        for c in categories:
            try:
                agg_stats[c] = agg_stats[c] + tile_stats[tile_id][c]
            except:
                continue

    print(agg_stats)

    return tile_stats, agg_stats


# DRAWING

def plot_image(array,figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    plt.imshow(array)
    plt.show()

def crop_raster(cutline, input, output):
    command = 'gdalwarp -q -cutline {0} -of GTiff {1} {2}'.format(cutline, input, output)
    print('>>>',command)
    try:
        s=0
        print(subprocess.check_output(command.split(), shell=False))
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    return

def crop_maps(cutline, inputs):
    outputs = []
    for input in inputs:
        output = input[0:input.index('.tif')] + '_cut.tif'
        outputs.append(output)
        #print(input, '->', output)
        crop_raster(cutline, input, output)
    return outputs
