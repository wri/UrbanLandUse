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
    geotrans = obj.GetGeoTransform()
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
    outds.SetGeoTransform(geotrans)
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


# RASTER CALCULATION & SAMPLING

def spectral_index(img,a,b,tol=1e-6):
    # returns (a - b)/(a + b)
    a_minus_b = np.add(img[:,:,a],np.multiply(img[:,:,b],-1.0))
    a_plus_b = np.add(np.add(img[:,:,a],img[:,:,b]),tol)
    y = np.divide(a_minus_b,a_plus_b)
    y = np.clip(y,-1.0,1.0)
    a_minus_b = None
    a_plus_b = None
    return y

def spectral_index_tile(img,a,b,tol=1e-6):
    # returns (a - b)/(a + b)
    a_minus_b = np.add(img[a,:,:],np.multiply(img[b,:,:],-1.0))
    a_plus_b = np.add(np.add(img[a,:,:],img[b,:,:]),tol)
    y = np.divide(a_minus_b,a_plus_b)
    y = np.clip(y,-1.0,1.0)
    a_minus_b = None
    a_plus_b = None
    return y

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

def ls_haze_removal(img,nodata,thresh=2):
    #
    # based on https://github.com/descarteslabs/hedj/blob/master/crops/ml_exp/lacid.py
    #
    refl_min = np.array([ 0.04,  0.03,  0.03,  0.03,  0.03,  0.03])  # landsat [B,G,R,NIR,SWIR1,SWIR2]
    # refl_min = np.array([ 0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001])  # landsat [B,G,R,NIR,SWIR1,SWIR2]
    #
    n_bands = img.shape[0]
    n_pixels = img.shape[1]*img.shape[2]
    valid_fraction = (nodata==0).sum()/float(n_pixels)
    refl_offsets = np.zeros(n_bands,dtype=np.float32)
    corrected_img = np.zeros(img.shape,dtype=np.float32)
    #
    # calculate number of dark pixels
    nir = img[3]
    nir_min = refl_min[3]
    red = img[2]
    red_min = refl_min[2]
    # mask = (nir>nir_min) & ((nir<0.1) & (nodata==0))
    mask = (nir<0.1) & (red>red_min)
    n_dark_pix = np.sum(mask)
    #
    # check that we have enough pixels to work with
    if ((valid_fraction<0.01) | (n_dark_pix<100)):
        print "not enough dark pixels for haze correction"
        return img, refl_offsets
    #
    if n_dark_pix>1e5:
        ds = float(int(n_dark_pix/1e5))
    else:
        ds = 1.
    n_dark_pix /= ds
    print 'n_dark_pixels: %i' % n_dark_pix
    #
    # iterate over bands, measure haze offsets
    offsets = np.zeros(n_bands,dtype=np.float32)
    for b in range(len(refl_min)):
        #
        # no correction for SWIR
        if b>3:
            continue
        #
        # this_offset = np.percentile(img[b][mask][::ds,None], thresh)
        this_offset = np.percentile(img[b][mask], thresh)
        #
        # reflectance offsets should be monotonically decreasing
        if b>0:
            this_offset = np.min([this_offset, offsets[b-1]])
        offsets[b] = this_offset
    #
    # APPLY CORRECTION
    refl_offsets = offsets - refl_min
    refl_offsets[[4,5]] = 0.0    
    for b in range(n_bands):
        corrected_img[b][:,:] = img[b][:,:] - refl_offsets[b]
        corrected_img[b] = np.clip(corrected_img[b],0.0,1.5)
        print b, offsets[b], refl_offsets[b], img[b].min(), corrected_img[b].min(), corrected_img[b].max() 
    #
    return corrected_img, refl_offsets


# DRAWING

def plot_image(array,figsize=(6,6)):
    plt.imshow(array)
    plt.show()




# MASKING

    # CATEGORIZING

def cloud_mask_S2(X,get_rgb=False, 
                  key={0:'nodata',1:'land',2:'snow',3:'water',4:'clouds',5:'vegetation',6:'shadows'}):
    #
    #    clouds are bright, visibly gray, and relatively cool
    #
    #    inputs:
    #    X       6-band landsat images : VIS/NIR/SWIR bands[1,2,3,4,5,7] in top-of-atmosphere reflectance
    #
    #    output:
    #    Y       byte-valued cloud/snow/water/shadow mask
    # 
    #    vals:   (based on official NASA LTK cloud mask labels)
    #    1       land
    #    2       snow
    #    3       water bodies
    #    4       clouds
    #    5       vegetation
    #    6       shadows  (extra category not in the standard NASA label set)
    #
    L1 = X[:,:,0]
    L2 = X[:,:,1]
    L3 = X[:,:,2]
    L5 = X[:,:,4]
    alpha = X[:,:,-1]
    #
    # land - default value
    #
    Y = np.ones(L1.shape,dtype='uint8')
    #
    # snow/ice
    #
    ndsi = spectral_index(X,1,4)
    Y[(ndsi > 0.4)]  = 2  
    #
    ndvi = spectral_index(X,3,2)
    #
    # Y[(ndvi <= 0.0)] = 3  
    #
    # shadows
    #
    shadow_index = (ndvi <= 0.0)
    Y[shadow_index] = 6
    #
    # water
    #
    ratioBR = np.divide(L1,np.add(L3,1e-6))
    water_index = np.logical_and((ndvi <= 0.0),(ratioBR >= 1.0))
    Y[water_index] = 3  # water
    #
    # vegetation
    #
    Y[(ndvi > 0.2)] = 5
    #
    # CLOUD INDEX
    # 
    index = (L2 >= 0.20)  # 0.40
    #
    grey = spectral_index(X,1,0)
    index = np.logical_and(index, (abs(grey) < 0.2))    
    grey = spectral_index(X,2,1)
    index = np.logical_and(index, (abs(grey) < 0.2))
    Y[index] = 4  # clouds
    #
    # nodata 
    #
    Y[(alpha==0)]=0
    #
    if (get_rgb==True):
        rgb = rgb_clouds(Y)
        return Y,rgb,key
    #
    return Y,key

def cloud_mask_S2_tile(X,get_rgb=False, 
                  key={0:'nodata',1:'land',2:'snow',3:'water',4:'clouds',5:'vegetation',6:'shadows'}):
    #
    #    clouds are bright, visibly gray, and relatively cool
    #
    #    inputs:
    #    X       6-band landsat images : VIS/NIR/SWIR bands[1,2,3,4,5,7] in top-of-atmosphere reflectance
    #
    #    output:
    #    Y       byte-valued cloud/snow/water/shadow mask
    # 
    #    vals:   (based on official NASA LTK cloud mask labels)
    #    1       land
    #    2       snow
    #    3       water bodies
    #    4       clouds
    #    5       vegetation
    #    6       shadows  (extra category not in the standard NASA label set)
    #
    L1 = X[0,:,:]
    L2 = X[1,:,:]
    L3 = X[2,:,:]
    L5 = X[4,:,:]
    alpha = X[-1,:,:]
    #
    # land - default value
    #
    Y = np.ones(L1.shape,dtype='uint8')
    #
    # snow/ice
    #
    ndsi = spectral_index_tile(X,1,4)
    Y[(ndsi > 0.4)]  = 2  
    #
    ndvi = spectral_index_tile(X,3,2)
    #
    # Y[(ndvi <= 0.0)] = 3  
    #
    # shadows
    #
    shadow_index = (ndvi <= 0.0)
    Y[shadow_index] = 6
    #
    # water
    #
    ratioBR = np.divide(L1,np.add(L3,1e-6))
    water_index = np.logical_and((ndvi <= 0.0),(ratioBR >= 1.0))
    Y[water_index] = 3  # water
    #
    # vegetation
    #
    Y[(ndvi > 0.2)] = 5
    #
    # CLOUD INDEX
    # 
    index = (L2 >= 0.20)  # 0.40
    #
    grey = spectral_index_tile(X,1,0)
    index = np.logical_and(index, (abs(grey) < 0.2))    
    grey = spectral_index_tile(X,2,1)
    index = np.logical_and(index, (abs(grey) < 0.2))
    Y[index] = 4  # clouds
    #
    # nodata 
    #
    Y[(alpha==0)]=0
    #
    if (get_rgb==True):
        rgb = rgb_clouds(Y)
        return Y,rgb,key
    #
    return Y,key


def cloud_mask(X,T,get_rgb=False, 
               key={0:'nodata',1:'land',2:'snow',3:'water',4:'clouds',5:'vegetation',6:'shadows'}):
    #
    #    clouds are bright, visibly gray, and relatively cool
    #
    #    inputs:
    #    X       6-band landsat images : VIS/NIR/SWIR bands[1,2,3,4,5,7] in top-of-atmosphere reflectance
    #    T       1-band landsat Thermal IR band : at-sensor temperature in celsius in range [-32.0C, 96.0C]
    #
    #    output:
    #    Y       byte-valued cloud/snow/water/shadow mask
    # 
    #    vals:   (based on official NASA LTK cloud mask labels)
    #    1       land
    #    2       snow
    #    3       water bodies
    #    4       clouds
    #    5       vegetation
    #    6       shadows  (extra category not in the standard NASA label set)
    #
    L1 = X[:,:,0]
    L2 = X[:,:,1]
    L3 = X[:,:,2]
    L5 = X[:,:,4]
    L6 = T   # celsius
    alpha = X[:,:,-1]
    #
    # land - default value
    #
    Y = np.ones(L1.shape,dtype='uint8')
    #
    # snow/ice
    #
    ndsi = spectral_index(X,1,4)
    #
    # Y[(L6 <= 0)] = 2  # v2
    Y[np.logical_or((L6 <= 0),(ndsi > 0.4))]  = 2  
    #
    ndvi = spectral_index(X,3,2)
    #
    # Y[(ndvi <= 0.0)] = 3  
    #
    # shadows
    #
    shadow_index = np.logical_and((ndvi <= 0.0),(L6 > 0))
    Y[shadow_index] = 6
    #
    # water
    #
    ratioBR = np.divide(L1,np.add(L3,1e-6))
    # water_index = np.logical_and(shadow_index,(ratioBR > 1.0))
    water_index = np.logical_and((ndvi <= 0.0),(ratioBR >= 1.0))
    # ratioBG = np.divide(L1,np.add(L2,1e-8))
    # water_index = np.logical_and(shadow_index,(ratioBG > 1.0))
    # water_index = (ndvi <= 0.0) # simplest
    Y[water_index] = 3  # water
    #
    # vegetation
    #
    Y[(ndvi > 0.2)] = 5
    #
    # CLOUD INDEX
    # 
    # index = (L5 > 0.30)  # 0.40
    # index = (L1 > 0.20)  # 0.40
    index = (L2 >= 0.20)  # 0.40
    #
    grey = spectral_index(X,1,0)
    index = np.logical_and(index, (abs(grey) < 0.2))    
    grey = spectral_index(X,2,1)
    index = np.logical_and(index, (abs(grey) < 0.2))
    #
    # index = np.logical_and(index, (L6 < 15))  # 280
    index = np.logical_and(index, (L6 < 30))  # 280
    #
    Y[index] = 4  # clouds
    #
    # nodata 
    #
    Y[(alpha==0)]=0
    #
    if (get_rgb==True):
        rgb = rgb_clouds(Y)
        return Y,rgb,key
    #
    return Y,key


# COMPOSITING

# calculations from imagery dynamically acquired through dl api
# references to this should be replaced with references to calc_index_minmax
def calc_ndvi_minmax(s2_ids, tiles, shape, resampler='bilinear'):
    bands=['blue','green','red','nir','swir1','swir2','alpha'];

    ntiles = len(tiles['features'])
    tilepixels = tiles['features'][0]['properties']['tilesize'] + (tiles['features'][0]['properties']['pad']*2)

    max_tiles = np.full([ntiles, tilepixels, tilepixels], np.nan, dtype='float32')
    min_tiles = np.full([ntiles, tilepixels, tilepixels], np.nan, dtype='float32')

    for j in range(len(s2_ids)):
        print 'image', j
        for tile_id in range(ntiles):
            if(tile_id % 50 == 0):
               print '    tile', tile_id
            tile = tiles['features'][tile_id]

            try:
                vir, vir_meta = dl.raster.ndarray(
                    s2_ids[j],
                    bands=bands,
                    data_type='UInt16',
                    dltile=tile,
                    resampler=resampler,
                    #cutline=shape['geometry'] # removing to try to sidestep nan issue
                )
            except Exception as e:
                print 'ProtocolError when acquiring tile #'+str(tile_id)+' for image #'+str(j)+' ('+str(s2_ids[j])+'):'
                print e
                import time
                time.sleep(10)
                tile_id-=1
                continue

            #print vir.shape
            vir = vir.astype('float32')
            vir = vir/10000.
            vir = np.clip(vir,0.0,1.0)

            ndvi_j = spectral_index(vir,3,2)
            util_rasters_masking, key = cloud_mask_S2(vir)

            if (j==0):
                ndvi_max = np.full([ndvi_j.shape[0], ndvi_j.shape[1] ], np.nan, dtype=vir.dtype)
                ndvi_min = np.full([ndvi_j.shape[0], ndvi_j.shape[1] ], np.nan, dtype=vir.dtype)

            alpha_j = vir[:,:,-1]
            cloudfree_j = (util_rasters_masking != 4)
            goodpixels_j = np.logical_and(alpha_j, cloudfree_j)

            np.fmax(ndvi_j, max_tiles[tile_id], out=max_tiles[tile_id], where=goodpixels_j)
            np.fmin(ndvi_j, min_tiles[tile_id], out=min_tiles[tile_id], where=goodpixels_j)
    print 'done'
    return min_tiles, max_tiles


def calc_index_minmax(s2_ids, tiles, shape, band_a, band_b):
    bands=['blue','green','red','nir','swir1','swir2','alpha'];

    ntiles = len(tiles['features'])
    tilepixels = tiles['features'][0]['properties']['tilesize'] + (tiles['features'][0]['properties']['pad']*2)

    max_tiles = np.full([ntiles, tilepixels, tilepixels], np.nan, dtype='float32')
    min_tiles = np.full([ntiles, tilepixels, tilepixels], np.nan, dtype='float32')

    for j in range(len(s2_ids)):
        print 'image', j
        for tile_id in range(ntiles):
            if(tile_id % 50 == 0):
               print '    tile', tile_id
            tile = tiles['features'][tile_id]

            vir, vir_meta = dl.raster.ndarray(
                s2_ids[j],
                bands=bands,
                data_type='UInt16',
                dltile=tile,
                #cutline=shape['geometry'] # removing to try to sidestep nan issue
            )
            #print vir.shape
            vir = vir.astype('float32')
            vir = vir/10000.
            vir = np.clip(vir,0.0,1.0)

            ndvi_j = bronco.spectral_index(vir,band_a,band_b)
            bronco_masking, key = bronco.cloud_mask_S2(vir)

            if (j==0):
                ndvi_max = np.full([ndvi_j.shape[0], ndvi_j.shape[1] ], np.nan, dtype=vir.dtype)
                ndvi_min = np.full([ndvi_j.shape[0], ndvi_j.shape[1] ], np.nan, dtype=vir.dtype)

            alpha_j = vir[:,:,-1]
            cloudfree_j = (bronco_masking != 4)
            goodpixels_j = np.logical_and(alpha_j, cloudfree_j)

            np.fmax(ndvi_j, max_tiles[tile_id], out=max_tiles[tile_id], where=goodpixels_j)
            np.fmin(ndvi_j, min_tiles[tile_id], out=min_tiles[tile_id], where=goodpixels_j)

    print 'done'
    return min_tiles, max_tiles




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

def show_vir_s2(file):
    img, geo, prj, cols, rows = load_geotiff(file)

    img = np.transpose(img, (1,2,0))
    img = img[:,:,0:3]
    img = np.flip(img, axis=-1)

    viz = img.astype('float32')
    viz = viz/3000
    viz = 255.*viz
    viz = np.clip(viz,0,255)
    viz = viz.astype('uint8')
    for b in range(img.shape[2]):
        print b, np.min(viz[:,:,b]), np.max(viz[:,:,b])
    plt.figure(figsize=[16,16])
    plt.imshow(viz)

def calc_water_mask(vir, idx_green=1, idx_nir=3, threshold=0.15):

    assert vir.shape[0]==6

    band_a = idx_green
    band_b = idx_nir
    threshold = threshold # water = ndwi > threshold 
    ndwi = spectral_index_tile(vir, band_a, band_b)
    water = ndwi > threshold

    return water

def make_water_mask_tile(data_path, place, tile_id, tiles, image_suffix, threshold):
    assert type(tile_id) is int 
    assert tile_id < len(tiles['features'])
    tile = tiles['features'][tile_id]
    resolution = int(tile['properties']['resolution'])

    #print 'tile', tile_id, 'load VIR image'
    # assert resolution==10 or resolution==5
    if resolution==10:
        zfill = 3
    elif resolution==5:
        zfill = 4
    elif resolution==2:
        zfill=5
    else:
        raise Exception('bad resolution: '+str(resolution))
    vir_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_vir_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

    #print vir_file
    vir, virgeo, virprj, vircols, virrows = load_geotiff(vir_file,dtype='uint16')
    #print 'vir shape:',vir.shape

    water = calc_water_mask(vir[0:6], threshold=threshold)
    water_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_water_'+image_suffix+'.tif'
    print water_file
    write_1band_geotiff(water_file, water, virgeo, virprj, data_type=gdal.GDT_Byte)
    return water

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