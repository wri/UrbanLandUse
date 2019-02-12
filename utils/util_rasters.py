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
import bronco

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
        #print outfile, cols, rows, bands, data_type
        outds  = driver.Create(outfile, cols, rows, bands, data_type)
    outds.SetGeoTransform(geotrans)
    outds.SetProjection(prj)
    for b in range(bands):
        outds.GetRasterBand(b+1).WriteArray(img[b])
    del outds

# assumes first dimension is bands
def window(x,j,i,r,bands_first=True):
    if bands_first:
        w = x[:,j-r:j+r+1,i-r:i+r+1]
    else:
        w = x[j-r:j+r+1,i-r:i+r+1,:]
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
        lulc=True, show=False, band_index=0):
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
            print c, yd[c], category_label[c] if c in category_label else ''
    if(show):
        if lulc:
            rgb = rgb_lulc_result(y)
            plt.figure(figsize=[8,8])
            plt.imshow(rgb)
        else:
            y2 = y.copy()
            plt.figure(figsize=[8,8])
            plt.imshow(y2)
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
            print tile_id, tile_stats[tile_id]
        for c in categories:
            try:
                agg_stats[c] = agg_stats[c] + tile_stats[tile_id][c]
            except:
                continue

    print agg_stats

    return tile_stats, agg_stats


# DRAWING

def plot_image(array,figsize=(6,6)):
    plt.imshow(array)
    plt.show()

    # RGP REMAPPING

def rgb_clouds(Y,BIP=True):
    rgb = np.zeros((3,Y.shape[0],Y.shape[1]),dtype='uint8')
    # nodata
    rgb[0][(Y==0)] = 255
    rgb[1][(Y==0)] = 255
    rgb[2][(Y==0)] = 255
    # non-vegetated land -> Tan (210,180,140)
    rgb[0][(Y==1)] = 210
    rgb[1][(Y==1)] = 180
    rgb[2][(Y==1)] = 140
    # snow/ice -> (220,220,220)
    rgb[0][(Y==2)] = 220
    rgb[1][(Y==2)] = 220
    rgb[2][(Y==2)] = 220
    # water bodies -> lighter (119,214,232), or darker (98,176,214)
    rgb[0][(Y==3)] =  98  # 119
    rgb[1][(Y==3)] = 176  # 214
    rgb[2][(Y==3)] = 214  # 232
    # clouds -> (241,248,251)
    rgb[0][(Y==4)] = 241
    rgb[1][(Y==4)] = 248
    rgb[2][(Y==4)] = 251
    # vegetation -> lighter (120,176,21), or darker (134,148,21)
    rgb[0][(Y==5)] = 134  # 120
    rgb[1][(Y==5)] = 148  # 176
    rgb[2][(Y==5)] =  21  #  21
    # shadow -> (64,64,64)
    rgb[0][(Y==6)] = 128 
    rgb[1][(Y==6)] = 128
    rgb[2][(Y==6)] = 128
    #
    if (BIP==True):
        tmp = np.zeros((Y.shape[0],Y.shape[1],3),dtype='uint8')
        for b in range(3):
            tmp[:,:,b] = rgb[b][:,:]
        rgb = tmp
    return rgb

def rgb_lulc_result(Y,BIP=True):
    rgb = np.zeros((3,Y.shape[0],Y.shape[1]),dtype='uint8')
    # open space
    rgb[0][(Y==0)] = int("b2", 16)
    rgb[1][(Y==0)] = int("df", 16)
    rgb[2][(Y==0)] = int("8a", 16)
    # non-residential
    rgb[0][(Y==1)] = int("fb", 16)
    rgb[1][(Y==1)] = int("9a", 16)
    rgb[2][(Y==1)] = int("99", 16)
    # residential - atomistic
    rgb[0][(Y==2)] = int("ff", 16)
    rgb[1][(Y==2)] = int("7f", 16)
    rgb[2][(Y==2)] = int("00", 16)
    # residential - informal
    rgb[0][(Y==3)] = int("fd", 16)
    rgb[1][(Y==3)] = int("bf", 16)
    rgb[2][(Y==3)] = int("6f", 16)
    # residential - formal
    rgb[0][(Y==4)] = int("1f", 16)
    rgb[1][(Y==4)] = int("78", 16)
    rgb[2][(Y==4)] = int("b4", 16)
    # residential - projects
    rgb[0][(Y==5)] = int("a6", 16)
    rgb[1][(Y==5)] = int("ce", 16)
    rgb[2][(Y==5)] = int("e3", 16)
    # roads
    rgb[0][(Y==6)] = int("e3", 16)
    rgb[1][(Y==6)] = int("1a", 16)
    rgb[2][(Y==6)] = int("1c", 16)
    # water
    rgb[0][(Y==9)] = int("00", 16)
    rgb[1][(Y==9)] = int("33", 16)
    rgb[2][(Y==9)] = int("66", 16)
    # outside study area
    rgb[0][(Y==254)] = int("00", 16)
    rgb[1][(Y==254)] = int("00", 16)
    rgb[2][(Y==254)] = int("00", 16)
    # no data
    rgb[0][(Y==255)] = int("f3", 16)
    rgb[1][(Y==255)] = int("f3", 16)
    rgb[2][(Y==255)] = int("f3", 16)
    #
    if (BIP==True):
        tmp = np.zeros((Y.shape[0],Y.shape[1],3),dtype='uint8')
        for b in range(3):
            tmp[:,:,b] = rgb[b][:,:]
        rgb = tmp
    return rgb

def rgb_esa_lulc(Y,BIP=True):
    # NB_LAB;LCCOwnLabel;R;G;B
    # 0;No data;0;0;0
    # 1;Tree cover areas;0;160;0
    # 2;Shrubs cover areas;150;100;0
    # 3;Grassland;255;180;0
    # 4;Cropland;255;255;100
    # 5;Vegetation aquatic or regularly flooded;0;220;130
    # 6;Lichens Mosses / Sparse vegetation;255;235;175
    # 7;Bare areas;255;245;215
    # 8;Built up areas;195;20;0
    # 9;Snow and/or Ice;255;255;255
    # 10;Open Water;0;70;200
    #
    rgb = np.zeros((3,Y.shape[0],Y.shape[1]),dtype='uint8')
    # No data
    rgb[0][(Y==0)] = 0
    rgb[1][(Y==0)] = 0
    rgb[2][(Y==0)] = 0
    # Tree cover areas
    rgb[0][(Y==1)] = 0
    rgb[1][(Y==1)] = 160
    rgb[2][(Y==1)] = 0
    # Shrubs cover areas
    rgb[0][(Y==2)] = 150
    rgb[1][(Y==2)] = 100
    rgb[2][(Y==2)] = 0
    # Grassland
    rgb[0][(Y==3)] = 255
    rgb[1][(Y==3)] = 180
    rgb[2][(Y==3)] = 0
    # Cropland
    rgb[0][(Y==4)] = 255
    rgb[1][(Y==4)] = 255
    rgb[2][(Y==4)] = 100
    # Vegetation aquatic or regularly flooded
    rgb[0][(Y==5)] = 0
    rgb[1][(Y==5)] = 220
    rgb[2][(Y==5)] = 130
    # Lichens Mosses / Sparse vegetation
    rgb[0][(Y==6)] = 255
    rgb[1][(Y==6)] = 235
    rgb[2][(Y==6)] = 175
    # Bare areas
    rgb[0][(Y==7)] = 255
    rgb[1][(Y==7)] = 245
    rgb[2][(Y==7)] = 215
    # Built up areas 
    rgb[0][(Y==8)] = 195
    rgb[1][(Y==8)] = 20
    rgb[2][(Y==8)] = 0
    # Snow and/or Ice
    rgb[0][(Y==9)] = 255
    rgb[1][(Y==9)] = 255
    rgb[2][(Y==9)] = 255
    # Open Water
    rgb[0][(Y==10)] = 0
    rgb[1][(Y==10)] = 70
    rgb[2][(Y==10)] = 200
    #
    if (BIP==True):
        tmp = np.zeros((Y.shape[0],Y.shape[1],3),dtype='uint8')
        for b in range(3):
            tmp[:,:,b] = rgb[b][:,:]
        rgb = tmp
    return rgb

def crop_raster(cutline, input, output):
    command = 'gdalwarp -q -cutline {0} -of GTiff {1} {2}'.format(cutline, input, output)
    print '>>>',command
    try:
        s=0
        print subprocess.check_output(command.split(), shell=False)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    return

def crop_maps(cutline, inputs):
    outputs = []
    for input in inputs:
        output = input[0:input.index('.tif')] + '_cut.tif'
        outputs.append(output)
        #print input, '->', output
        crop_raster(cutline, input, output)
    return outputs