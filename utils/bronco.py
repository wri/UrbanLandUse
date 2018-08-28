import warnings
warnings.filterwarnings('ignore')
#
import os
import sys
import json
import itertools
import pickle
from pprint import pprint
#
import numpy as np
import shapely
import cartopy
from osgeo import gdal
import matplotlib.pyplot as plt
#
import descarteslabs as dl

def load_geotiff(tif,dtype='uint16'):
    obj = gdal.Open(tif, gdal.gdalconst.GA_ReadOnly)
    prj = obj.GetProjection()
    geotrans = obj.GetGeoTransform()
    cols = obj.RasterXSize
    rows = obj.RasterYSize
    img = np.zeros((rows,cols), dtype=dtype)
    img = obj.ReadAsArray().astype(dtype)
    del obj
    return img, geotrans, prj, cols, rows

def write_1band_geotiff(outfile, img, geotrans, prj, data_type=gdal.GDT_Byte):
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

def spectral_index(img,a,b,tol=1e-6):
    # returns (a - b)/(a + b)
    a_minus_b = np.add(img[:,:,a],np.multiply(img[:,:,b],-1.0))
    a_plus_b = np.add(np.add(img[:,:,a],img[:,:,b]),tol)
    y = np.divide(a_minus_b,a_plus_b)
    y = np.clip(y,-1.0,1.0)
    a_minus_b = None
    a_plus_b = None
    return y

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

    
def window(x,j,i,r):
    w = x[:,j-r:j+r+1,i-r:i+r+1]
    return w

def score(Yhat,Y,report=True):
    totals = np.zeros(4,dtype='uint16')
    tp = np.logical_and((Yhat==1), (Y==1)).sum()
    fp = np.logical_and((Yhat==0), (Y==1)).sum()
    fn = np.logical_and((Yhat==1), (Y==0)).sum()
    tn = np.logical_and((Yhat==0), (Y==0)).sum()
    totals[0] += tp
    totals[1] += fp
    totals[2] += fn
    totals[3] += tn
    if report==True:
        print "input shapes", Yhat.shape, Y.shape
        print "scores [tp, fp, fn, tn]", tp, fp, fn, tn,         "accuracy", round(100*((tp + tn)/float(tp + fp + fn + tn)),1),"%"


def confusion(Yhat,Y,categories):
    n_categories = len(categories)
    confusion = np.zeros((n_categories,n_categories),dtype='uint32')
    for j in range(n_categories):
        for i in range(n_categories):
            confusion[j,i] = np.sum(np.logical_and((Yhat==categories[i]),(Y==categories[j])))
            # print j,i, confusion[j,i], categories[j], categories[i] 
        print categories[j], np.sum(confusion[j,:])
    print confusion
    print confusion.sum(), confusion.trace(), confusion.trace()/float(confusion.sum())
    return confusion

def show_scene(ids, geom={}, resolution=60,
               bands=['red','green','blue','alpha'],
               scales=[[0,3000],[0,3000],[0,3000],None],
               figsize=[16,16], title=""):
    arr, meta = dl.raster.ndarray(
        ids,
        bands=bands,
        scales=scales,
        data_type='Byte',
        resolution=resolution,
        cutline=geom,
    )
    print ids, arr.shape
    # pprint(meta)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    plt.imshow(arr)
    return

def show_tile(ids, tile={}, resolution=60,
              bands=['red','green','blue','alpha'],
              scales=[[0,3000],[0,3000],[0,3000],None],
              figsize=[16,16],title=""):
    arr, meta = dl.raster.ndarray(
        ids,
        bands=bands,
        scales=scales,
        data_type='Byte',
        dltile=tile,
    )
    print ids, arr.shape
    # pprint(meta)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    plt.imshow(arr)
    return

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

def save_visualize_image(ids, tile={}, resolution=60, geom={},
                         bands=['red','green','blue','alpha'],
                         scales=[[0,3000],[0,3000],[0,3000],None],
                         filename='viz.tif'):
    arr, meta = dl.raster.raster(
        ids,
        bands=bands,
        scales=scales,
        data_type='Byte',
        resolution=resolution,
        cutline=geom,
        save=True,
        outfile_basename=filename
    )

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
    # outside study area
    rgb[0][(Y==254)] = int("00", 16)
    rgb[1][(Y==254)] = int("00", 16)
    rgb[2][(Y==254)] = int("00", 16)
    # no data
    rgb[0][(Y==255)] = int("ff", 16)
    rgb[1][(Y==255)] = int("ff", 16)
    rgb[2][(Y==255)] = int("ff", 16)
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