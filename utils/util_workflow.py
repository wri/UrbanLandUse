# organized set of helper functions
# drawn from bronco.py and bronco notebooks
# topic: building training data

import warnings
warnings.filterwarnings('ignore')
#
#import os
import sys
#import json
import itertools
import pickle
#from pprint import pprint
#
import numpy as np
#import shapely
#import cartopy
from osgeo import gdal
import matplotlib.pyplot as plt
import math

import datetime
import csv
#
import descarteslabs as dl
import utils.util_vectors as util_vectors
import utils.util_rasters as util_rasters
import utils.util_ml as util_ml
import utils.util_imagery as util_imagery

from tensorflow.keras.models import load_model





def build_stack_label(
        bands_vir=['blue','green','red','nir','swir1','swir2'],
        bands_sar=['vv','vh'],
        bands_ndvi=None,
        bands_ndbi=None,
        bands_osm=None,
        ):
    params = locals()
    #print(params)
    for k,v in params.items():
        if type(v) is list:
            for member in v:
                assert (type(member) is str)
        else:
            assert (v is None)
    feature_count = 0
    stack_label = ''
    if bands_vir is not None:
        feature_count += len(bands_vir)
        stack_label += 'vir+'
    if bands_sar is not None:
        feature_count += len(bands_sar)
        stack_label += 'sar+'
    if bands_ndvi is not None:
        feature_count += len(bands_ndvi)
        stack_label += 'ndvi+'
    if bands_ndbi is not None:
        feature_count += len(bands_ndbi)
        stack_label += 'ndbi+'
    if bands_osm is not None:
        feature_count += len(bands_osm)
        stack_label += 'osm+'

    if stack_label.endswith('+'):
        stack_label = stack_label[:-1]
    return stack_label, feature_count

def prepare_input_stack(data_path, place, tiles, stack_label, feature_count, 
        image_suffix, window, tile_id, 
        bands_vir=['blue','green','red','nir','swir1','swir2'],
        bands_sar=['vv','vh'], bands_ndvi=None, bands_ndbi=None, bands_osm=None,
        haze_removal=False,):
    tile = tiles['features'][tile_id]
    print(tile['properties'])
    side_length = tile['properties']['tilesize'] + tile['properties']['pad']*2

    resolution = int(tile['properties']['resolution'])

    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5
    else:
        raise Exception('bad resolution: '+str(resolution))

    imn = np.zeros((feature_count,side_length,side_length),dtype='float32')
    n_features = imn.shape[0] 

    print('tile', tile_id, 'load VIR image')

    vir_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_vir_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

    print(vir_file)
    vir, virgeo, virprj, vircols, virrows = util_rasters.load_geotiff(vir_file,dtype='uint16')
    print('vir shape:',vir.shape)
    vir = vir.astype('float32')
    vir = vir/10000.
    vir = np.clip(vir,0.0,1.0)
    #print('tile', tile_id, 'make data mask from vir alpha')
    mask = (vir[6][:,:] > 0)  # vir[6] is the alpha band in the image, takes values 0 and 65535
    nodata = (vir[6][:,:]==0)
    print(np.sum(mask), "study area within image")
    print(mask.shape[0] * mask.shape[1], "full extent of image")
    # haze removal
    if haze_removal:
        virc, virc_ro = util_rasters.ls_haze_removal(vir[:-1],nodata)
        print(virc_ro)
        vir[:-1] = virc[:]

    b_start = 0
    if bands_vir is not None:
        for b in range(vir.shape[0]-1):
            print('vir band',b,'into imn band',b_start+b,'(',np.min(vir[b,:,:]),'-',np.max(vir[b,:,:]),')')
            imn[b_start+b][:,:] = vir[b][:,:]
        b_start += vir.shape[0]-1

    if bands_sar is not None:
        print('tile', tile_id, 'load SAR image')
        
        sar_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_sar_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'


        sar, sargeo, sarprj, sarcols, sarrows = util_rasters.load_geotiff(sar_file,dtype='uint16')
        print('sar shape:',sar.shape)
        sar = sar.astype('float32')
        sar = sar/255.
        sar = np.clip(sar,0.0,1.0)
        for b in range(sar.shape[0]):
            print('sar band',b,'into imn band',b_start+b,'(',np.min(sar[b,:,:]),'-',np.max(sar[b,:,:]),')')
            imn[b_start+b][:,:] = sar[b][:,:]
        b_start += sar.shape[0]

    if bands_ndvi is not None:
        if 'raw' in bands_ndvi:
            #print('tile', tile_id, 'calculate NDVI raw')
            # returns (a - b)/(a + b)
            tol=1e-6
            a_minus_b = np.add(vir[3,:,:],np.multiply(vir[2,:,:],-1.0))
            a_plus_b = np.add(np.add(vir[3,:,:],vir[2,:,:]),tol)
            y = np.divide(a_minus_b,a_plus_b)
            y = np.clip(y,-1.0,1.0)
            a_minus_b = None
            a_plus_b = None
            ndvi_raw = y #nir, red
            print('ndvi_raw shape:', ndvi_raw.shape)
            print('ndvi raw into imn band',b_start,'(',np.min(ndvi_raw),'-',np.max(ndvi_raw),')')
            imn[b_start] = ndvi_raw
            b_start += 1
        if 'min' in bands_ndvi:
            print('tile', tile_id, 'load NDVI min')
            
            ndvi_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_ndvimin'+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

            ndvimin, ndvigeo, ndviprj, ndvicols, ndvirows = util_rasters.load_geotiff(ndvi_file,dtype='float32')
            if(np.sum(np.isnan(ndvimin)) > 0):
                ndvi_nan = np.isnan(ndvimin)
                print('nan ndvi inside study area:',np.sum(np.logical_and(ndvi_nan, mask)))
                ndvimin[ndvi_nan]=0
            print('ndvi min into imn band',b_start,'(',np.min(ndvimin),'-',np.max(ndvimin),')')
            imn[b_start] = ndvimin
            b_start += 1
        if 'max' in bands_ndvi:
            print('tile', tile_id, 'load NDVI max')
            
            ndvi_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_ndvimax'+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

            ndvimax, ndvigeo, ndviprj, ndvicols, ndvirows = util_rasters.load_geotiff(ndvi_file,dtype='float32')
            if(np.sum(np.isnan(ndvimax)) > 0):
                ndvi_nan = np.isnan(ndvimax)
                print('nan ndvi inside study area:',np.sum(np.logical_and(ndvi_nan, mask)))
                ndvimax[ndvi_nan]=0
            print('ndvi max into imn band',b_start,'(',np.min(ndvimax),'-',np.max(ndvimax),')')
            imn[b_start] = ndvimax
            b_start += 1

    if bands_ndbi is not None:
        if 'raw' in bands_ndbi:
            #print('tile', tile_id, 'calculate ndbi raw')
            # returns (a - b)/(a + b)
            tol=1e-6
            a_minus_b = np.add(vir[3,:,:],np.multiply(vir[2,:,:],-1.0))
            a_plus_b = np.add(np.add(vir[3,:,:],vir[2,:,:]),tol)
            y = np.divide(a_minus_b,a_plus_b)
            y = np.clip(y,-1.0,1.0)
            a_minus_b = None
            a_plus_b = None
            ndbi_raw = y #nir, red
            print('ndbi_raw shape:', ndbi_raw.shape)
            print('ndbi raw into imn band',b_start,'(',np.min(ndbi_raw),'-',np.max(ndbi_raw),')')
            imn[b_start] = ndbi_raw
            b_start += 1
        if 'min' in bands_ndbi:
            print('tile', tile_id, 'load ndbi min')
           
            ndbi_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_ndbimin'+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

            ndbimin, ndbigeo, ndbiprj, ndbicols, ndbirows = util_rasters.load_geotiff(ndbi_file,dtype='float32')
            if(np.sum(np.isnan(ndbimin)) > 0):
                ndbi_nan = np.isnan(ndbimin)
                print('nan ndbi inside study area:',np.sum(np.logical_and(ndbi_nan, mask)))
                ndbimin[ndbi_nan]=0
            print('ndbi min into imn band',b_start,'(',np.min(ndbimin),'-',np.max(ndbimin),')')
            imn[b_start] = ndbimin
            b_start += 1
        if 'max' in bands_ndbi:
            print('tile', tile_id, 'load ndbi max')

            ndbi_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_ndbimax'+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

            ndbimax, ndbigeo, ndbiprj, ndbicols, ndbirows = util_rasters.load_geotiff(ndbi_file,dtype='float32')
            if(np.sum(np.isnan(ndbimax)) > 0):
                ndbi_nan = np.isnan(ndbimax)
                print('nan ndbi inside study area:',np.sum(np.logical_and(ndbi_nan, mask)))
                ndbimax[ndbi_nan]=0
            print('ndbi max into imn band',b_start,'(',np.min(ndbimax),'-',np.max(ndbimax),')')
            imn[b_start] = ndbimax
            b_start += 1

    if bands_osm is not None:
        if 'roads' in bands_osm:
            print('tile', tile_id, 'load OSM roads')
            
            osm_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_osm'+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

            osm, osmgeo, osmprj, osmcols, osmrows = util_rasters.load_geotiff(osm_file,dtype='uint8')
            osm[osm==255] = 0
            osm = osm.astype('float32')
            osm = np.clip(osm,0.0,1.0)
            print('osm roads into imn band',b_start,'(',np.min(osm),'-',np.max(osm),')')
            imn[b_start] = osm
            b_start += 1

    print('imn', imn.shape, n_features)
    return mask, imn, virgeo, virprj

def prepare_output_stack(data_path, place, tiles, 
        label_suffix, mask, category_label, tile_id):
    
    b= tiles['features'][tile_id]['properties']['pad']
    print('tile', tile_id, 'load labels')
    resolution = int(tiles['features'][tile_id]['properties']['resolution'])
    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))
    label_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_'+label_suffix+('' if resolution==10 else ('_'+str(resolution))+'m')+'.tif'
    print(label_file)
    lb, lbgeo, lbprj, lbcols, lbrows = util_rasters.load_geotiff(label_file,dtype='uint8')
    #print("NYU AoUE labels", label_file, lbcols, lbrows, lbgeo, lbprj)
    # delete training points close to edge
    lb[0:b,:] = 255; lb[-b:,:] = 255
    lb[:,0:b] = 255; lb[:,-b:] = 255
    y = np.zeros((9,mask.shape[0],mask.shape[1]),dtype='byte')
    y[0] = (mask==1); y[0] &= (lb==0)
    y[1] = (lb==1)
    y[2] = (lb==2)
    y[3] = (lb==3)#; y[3] |= (lb==2)  # merge categories 2 and 3
    #y[4] = (lb==4); y[4] |= (lb==5)  # merge categories 4 and 5
    #change for 4-category typology that consolidates all residential types
    y[4] = (lb==4) #unmerging; can merge in memory once loaded     ; y[4] |= (lb==5); y[4] |= (lb==2); y[4] |= (lb==3) # merge categories 2,3,4,5
    y[5] = (lb==5)
    y[6] = (lb==6)
    y[7] = (mask==1)
    y[8] = (lb!=255)
    print('y.shape', y.shape)
    # remember that output here represents the consolidated categories (ie y[4] is more than just cat4)
    for i in range(9):
        print(i, np.sum(y[i]), category_label[i] )
    print('tile', tile_id, 'collect data,label samples')
    return y

def build_training_samples(data_path, place, stack_label, 
        image_suffix, label_suffix, window, imn, y, tile_id,
        categories=[0,1,2,3,4,5,6], resolution=5):

    r = window/2
    n_features = imn.shape[0] 

    n_samples = {}
    n_all_samples = 0
    for c in categories:
        n_samples[c] = np.sum(y[c])
        n_all_samples = n_all_samples + n_samples[c]
    print("n_samples, sum", n_samples, n_all_samples)
    X_data = np.zeros((n_all_samples,window*window*n_features),dtype=imn.dtype)  # imn2
    Y_data = np.zeros((n_all_samples),dtype='uint8')
    print("X,Y shapes", X_data.shape, Y_data.shape)
    index = 0
    for ki in range(len(categories)):
        k = categories[ki]
        n_k = np.sum((y[k] == 1))
        if (n_k==0):
            print('WARNING: tile', tile_id, 'category', ki, n_k, 'no training examples, continuing')
            continue
        print(k, categories[ki], n_k)
        z_k = np.where((y[k]==1))
        n_k = len(z_k[0])
        if n_k != n_samples[k]:
            print("error! mismatch",n_k, n_samples[k])
        X_k = np.zeros((window*window*n_features,n_k),imn.dtype)  # imn2
        for s in range(n_k):
            w = util_rasters.window(imn,z_k[0][s],z_k[1][s],r) # imn2
            X_k[:,s] = w.flatten()
        X_k = X_k.T

        X_k_nan = np.isnan(X_k)
        if(np.sum(X_k_nan) > 0):
            print('NaN in training data')
            print(np.where(X_k_nan))
        #perm = np.random.permutation(X_k.shape[0])
        #X_k = X_k[perm[:],:]
        Y_k = np.full((n_samples[k]), fill_value=k, dtype='uint8')
        X_data[index:index+n_samples[k],:] = X_k[:,:]
        Y_data[index:index+n_samples[k]] = Y_k[:]
        index = index + n_samples[k]
        print(k, index, X_k.shape, Y_k.shape)
    print(X_data.shape, Y_data.shape, X_data.dtype)

    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))

    if ((n_all_samples > 0) and (np.sum((y[0] == 1)) < 30000)):  # <<<< WARNING: HARD-WIRED LIMIT
        label_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'

        print(label_file)
        pickle.dump((X_data,Y_data), open(label_file, 'wb'))
    else:
        print('n_all_samples:', n_all_samples, 'mask true:', np.sum((y[0]==1)))
        print('WARNING: tile', tile_id, ' defective tile', n_all_samples, np.sum((y[0] == 1)))
    # del imn, X_data, Y_data
    print('tile', tile_id, 'done')
    print('') #line between tiles in output for readability

def construct_dataset_tiles(data_path, place, tiles, label_stats, image_suffix,
        window, stack_label, feature_count,
        bands_vir=['blue','green','red','nir','swir1','swir2'],
        bands_sar=['vv','vh'], bands_ndvi=None, bands_ndbi=None, bands_osm=None,
        haze_removal=False,
        label_suffix='labels', 
        category_label={0:'Open Space',1:'Non-Residential',\
                   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
                   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
                   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'} ):
    
    # note that method does not consistently utilize individual bands in the various lists
    # in order to construct data cube

    print("Feature count:", feature_count)
    print("Stack label: ", stack_label)

    if window/2 > tiles['features'][0]['properties']['pad']:
        raise ValueError("trying to use look window that exceeds size of available imagery tiles")
    
    resolution = int(tiles['features'][0]['properties']['resolution'])

    # fundamentally a tile-by-tile process
    for tile_id in range(len(tiles['features'])):
        # skip tiles without labels
        # HARDCODED THRESHOLD
        if (len(label_stats[tile_id].keys())==1) or (label_stats[tile_id][255]<40000):
            # print('WARNING: tile', tile_id, ' has no labels')
            continue
        
        mask, imn, geo, prj = prepare_input_stack(data_path, place, tiles, stack_label, feature_count, 
            image_suffix, window, tile_id, bands_vir=bands_vir, bands_sar=bands_sar, 
            bands_ndvi=bands_ndvi, bands_ndbi=bands_ndbi, bands_osm=bands_osm,
            haze_removal=False)


        y = prepare_output_stack(data_path, place, tiles, 
            label_suffix, mask, category_label, tile_id)
        
        build_training_samples(data_path, place, stack_label, 
            image_suffix, label_suffix, window, imn, y, tile_id, resolution=resolution)
        

def combine_dataset_tiles(data_path, place, tiles, label_suffix, image_suffix, stack_label, window,
        resolution=10, tile_min=None, tile_max=None):
    if tile_min is not None or tile_max is not None:
        assert tile_min is not None
        assert tile_max is not None
    n_samples = 0
    n_features = 0
    n_dtype = 'none'

    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))

    # for tile_id in [single_tile_id]:
    for tile_id in range(len(tiles['features'])):
        if tile_min is not None and tile_id < tile_min:
            continue
        if tile_max is not None and tile_id > tile_max:
            break
        label_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'


        print(label_file)

        try:
            with open(label_file, "rb") as f:
                Xt, Yt = pickle.load(f)
            f.close()
        except:
            #print('tile', str(tile_id).zfill(3), 'has no training samples')
            continue

        print(tile_id, Xt.shape, Yt.shape)
        n_samples = n_samples + Yt.shape[0]
        if (n_features==0):
            n_features = Xt.shape[1]
            n_dtype = Xt.dtype
        assert n_features==Xt.shape[1]
        assert n_dtype==Xt.dtype

    print(n_samples, n_features, n_dtype)

    X_data = np.zeros((n_samples,n_features),dtype=n_dtype)
    Y_data = np.zeros((n_samples),dtype='uint8')

    print(X_data.shape, Y_data.shape)

    n_start = 0
    # for tile_id in [single_tile_id]:
    for tile_id in range(len(tiles['features'])):
        if tile_min is not None and tile_id < tile_min:
            continue
        if tile_max is not None and tile_id > tile_max:
            break

        label_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'

        # print(label_file)

        try:
            with open(label_file, "rb") as f:
                Xt, Yt = pickle.load(f)
            f.close()
        except:
            #print('tile', str(tile_id).zfill(3), 'has no training samples')
            continue

        # print(tile_id, Xt.shape, Yt.shape)
        n_t = Yt.shape[0]
        n_end = n_start + n_t
        X_data[n_start:n_end,:] = Xt[:,:]
        Y_data[n_start:n_end] = Yt[:]
        print(n_start, n_end)
        n_start = n_end
    print(X_data.shape, Y_data.shape)
    if tile_min is None:
        data_file = data_path+place+'_data_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
    else:
        data_file = data_path+place+'_data_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'_tile'+str(tile_min)+'-'+str(tile_max)+'.pkl'
    #print('Write complete datasets to file:', data_file)
    pickle.dump((X_data,Y_data), open(data_file, 'wb'))

    return X_data, Y_data

def split_dataset(data_path, place, label_suffix, stack_label, image_suffix, window, 
    resolution=10, tile_min=None, tile_max=None):
    if tile_min is not None or tile_max is not None:
        assert tile_min is not None
        assert tile_max is not None
        data_file = data_path+place+'_data_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'_tile'+str(tile_min)+'-'+str(tile_max)+'.pkl'
    else:
        data_file = data_path+place+'_data_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
    print(data_file)
    with open(data_file, "rb") as f:
        X_data, Y_data = pickle.load(f)
    f.close()
    
    n_samples = Y_data.shape[0]
    
    perm_file = data_path+place+'_perm_'+label_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
    print(perm_file)
    try:
        with open(perm_file, "rb") as f:
            perm = pickle.load(f)
    except IOError as e:
        print('Unable to open file:', perm_file) #Does not exist OR no read permissions
        print('Create permutation of length', str(n_samples))
        perm = np.random.permutation(n_samples)
        pickle.dump((perm), open(perm_file, 'wb'))
        
    print(len(perm), perm)

    if Y_data.shape[0] != perm.shape[0]:
        print('Cannot use indicated permutation to generate training and validation files from data file:', data_file)
        print('permutation has shape', perm.shape)
        print('X_data has shape', X_data.shape)
        return

        
    X_data = X_data[perm[:],:]
    Y_data = Y_data[perm[:]]

    data_scale = 1.
    X_data = X_data/data_scale

    n_train = int(math.floor(0.70*n_samples))
    n_valid = n_samples - n_train
    print(n_samples, n_train, n_valid)

    X_train = X_data[:n_train,:]
    X_valid = X_data[n_train:,:]

    Y_train = Y_data[:n_train]
    Y_valid = Y_data[n_train:]

    print(X_train.shape, Y_train.shape)
    print(X_valid.shape, Y_valid.shape)

    train_file = data_path+place+'_train_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
    pickle.dump((X_train,Y_train), open(train_file, 'wb'))
    valid_file = data_path+place+'_valid_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
    pickle.dump((X_valid,Y_valid), open(valid_file, 'wb'))



def load_datasets(place_images, data_root, label_suffix, stack_label, window, resolution=10):

    print('calculate total size of training and validation supersets')
    t_total = 0
    v_total = 0
    for city, suffixes in place_images.items():
        for suffix in suffixes:
            train_file = data_root+city+'/'+city+'_train_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(train_file)
            with open(train_file, 'rb') as f:
                X_train_sub, Y_train_sub = pickle.load(f)
            f.close()
            valid_file = data_root+city+'/'+city+'_valid_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(valid_file)
            with open(valid_file, 'rb') as f:
                X_valid_sub, Y_valid_sub = pickle.load(f)
            f.close()
            print(X_train_sub.shape, Y_train_sub.shape, X_valid_sub.shape, Y_valid_sub.shape)
            t_sub = X_train_sub.shape[0]
            v_sub = X_valid_sub.shape[0]
            #print(t_sub, v_sub)
            t_total += t_sub
            v_total += v_sub
    print(t_total, v_total)

    print('construct np arrays for supersets')
    X_train = np.zeros((t_total, X_train_sub.shape[1]), dtype=X_train_sub.dtype)
    Y_train = np.zeros((t_total), dtype=Y_train_sub.dtype)
    X_valid = np.zeros((v_total, X_valid_sub.shape[1]), dtype=X_valid_sub.dtype)
    Y_valid = np.zeros((v_total), dtype=Y_valid_sub.dtype)

    print('populate superset np arrays')
    v_start = 0
    t_start = 0
    for city, suffixes in place_images.items():
        for suffix in suffixes:
            train_file = data_root+city+'/'+city+'_train_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(train_file)
            with open(train_file, 'rb') as f:
                X_train_sub, Y_train_sub = pickle.load(f)
            f.close()
            valid_file = data_root+city+'/'+city+'_valid_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(valid_file)
            with open(valid_file, 'rb') as f:
                X_valid_sub, Y_valid_sub = pickle.load(f)
            f.close()
            print(X_train_sub.shape, Y_train_sub.shape, X_valid_sub.shape, Y_valid_sub.shape)
            t_sub = X_train_sub.shape[0]
            v_sub = X_valid_sub.shape[0]
            
            X_train[t_start:t_start+t_sub,:] = X_train_sub[:,:]
            Y_train[t_start:t_start+t_sub] = Y_train_sub[:]
            X_valid[v_start:v_start+v_sub,:] = X_valid_sub[:,:]
            Y_valid[v_start:v_start+v_sub] = Y_valid_sub[:]

            t_start = t_start + t_sub
            v_start = v_start + v_sub
    print(X_train.shape, Y_train.shape)
    return X_train, Y_train, X_valid, Y_valid

def create_classification_arrays(window, n_cats, imn, pad):
    r = window/2
    Y = np.zeros((imn.shape[1],imn.shape[2]),dtype='uint8')
    Y_deep = np.empty((imn.shape[1],imn.shape[2],n_cats),dtype='float32')
    Y_deep[:] = np.nan
    Y_max = np.empty((imn.shape[1],imn.shape[2]),dtype='float32')
    Y_max[:] = np.nan
    print("imn.shape, Y.shape", imn.shape, Y.shape)
    Y[:,:] = 255  # inside study area
    #Y_deep[:,:,:] = -1.0
    # buffer edge
    buff = max(r, pad)
    Y[0:buff,:]=254; Y[-buff:,:]=254; Y[:,0:buff]=254; Y[:,-buff:]=254
    z = np.where((Y==255))
    Y_deep[z[0][:],z[1][:],:] = 0.0
    Y_max[z[0][:],z[1][:]] = 0.0
    return Y, Y_deep, Y_max

def fill_classification_arrays(feature_count, window, n_cats, scaler, network, imn, Y, Y_deep, Y_max, unflatten_input=False, water_overwrite=False):
    data_scale = 1.0
    r = window/2
    z = np.where((Y==255))
    nz = len(z[0])
    print("nz", nz)
    cmax = 20
    nc = nz/cmax
    for c in range(cmax):
        j_c = z[0][c*nc:(c+1)*nc]
        i_c = z[1][c*nc:(c+1)*nc]
        X_c = np.zeros((window*window*feature_count,nc),imn.dtype)  # imn2
        for b in range(feature_count):
            for j in range(window):
                for i in range(window):
                    X_c[window*window*b + window*j + i,:] = imn[b,j_c[:]+j-r,i_c[:]+i-r]  # imn2
        X_c = X_c.T
        X_c = X_c/data_scale
        # X_c_scaled = X_c  
        X_c_scaled = scaler.transform(X_c)
        if unflatten_input:
            X_c_final = X_c_scaled.reshape((X_c_scaled.shape[0],feature_count,window,window))
        else:
            X_c_final = X_c_scaled
        Yhat_c_prob = network.predict(X_c_final)
        if n_cats==2:
            Yhat_c = np.rint(Yhat_c_prob).astype(int).reshape(Yhat_c_prob.shape[0])
            Yhat_max = Yhat_c.copy()
        else:
            Yhat_c = Yhat_c_prob.argmax(axis=-1)
            Yhat_max = np.amax(Yhat_c_prob,axis=-1)
            
        #set_trace()
        sys.stdout.write('.')
        Y[j_c[:],i_c[:]] = Yhat_c[:]
        Y_deep[j_c[:],i_c[:]] = Yhat_c_prob[:]
        Y_max[j_c[:],i_c[:]] = Yhat_max[:]
        #print('Category', Yhat_c)
        #print('Probability',Yhat_c_prob)
        #print('Maximum',Yhat_max)
        #print()
    for c in range(cmax,cmax+1):
        j_c = z[0][c*nc:]
        i_c = z[1][c*nc:]
        remainder = nz - c*nc
        X_c = np.zeros((window*window*feature_count,remainder),imn.dtype)  # imn2
        for b in range(feature_count):
            for j in range(window):
                for i in range(window):
                    X_c[window*window*b + window*j + i,:] = imn[b,j_c[:]+j-r,i_c[:]+i-r]  # imn2
        X_c = X_c.T
        X_c = X_c/data_scale
        # X_c_scaled = X_c  
        X_c_scaled = scaler.transform(X_c)
        if unflatten_input:
            X_c_final = X_c_scaled.reshape((X_c_scaled.shape[0],feature_count,window,window))
        else:
            X_c_final = X_c_scaled
        Yhat_c_prob = network.predict(X_c_final)
        if n_cats==2:
            Yhat_c = np.rint(Yhat_c_prob).astype(int).reshape(Yhat_c_prob.shape[0])
            Yhat_max = Yhat_c.copy()
        else:
            Yhat_c = Yhat_c_prob.argmax(axis=-1)
            Yhat_max = np.amax(Yhat_c_prob,axis=-1)
        #set_trace()
        sys.stdout.write('.')
        Y[j_c[:],i_c[:]] = Yhat_c[:]
        Y_deep[j_c[:],i_c[:]] = Yhat_c_prob[:]
        Y_max[j_c[:],i_c[:]] = Yhat_max[:]

    if water_overwrite:
        water = util_rasters.calc_water_mask(imn[0:6])
        cat_water = 9 # addition to 0-6 AUE taxonomy
        Y[water] = cat_water
    
    print("done")
    for k in range(255):
        if np.sum((Y==k))>0:
            print(k, np.sum((Y==k)))

def create_training_data(data_root, place_images, tile_resolution, tile_size, tile_pad, window, 
        bands_vir=['blue','green','red','nir','swir1','swir2'],
        bands_sar=['vv','vh'], bands_ndvi=None, bands_ndbi=None, bands_osm=None,
        haze_removal=False,
        label_suffix='labels', 
        category_label={0:'Open Space',1:'Non-Residential',\
                   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
                   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
                   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'} ):
    stack_label, feature_count = build_stack_label(
            bands_vir=bands_vir,
            bands_sar=bands_sar,
            bands_ndvi=bands_ndvi,
            bands_ndbi=bands_ndbi,
            bands_osm=bands_osm,)

    resolution = tile_resolution
    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))


    for place, image_suffix_list in place_images.items():
        data_path = data_root + place + '/'
        place_shapefile = data_path+place.title()+"_studyAreaEPSG4326.shp"
        shape = util_vectors.load_shape(place_shapefile)
        tiles = dl.raster.dltiles_from_shape(tile_resolution, tile_size, tile_pad, shape)
        label_stats = {}
        for tile_id in range(len(tiles['features'])):
            tile = tiles['features'][tile_id]
            label_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_'+label_suffix+'_'+('' if tile_resolution==10 else str(int(tile_resolution))+'m')+'.tif'
            label_stats[tile_id] = util_rasters.stats_byte_raster(label_file, category_label, show=False)

        for image_suffix in image_suffix_list:
            print('Constructing dataset tiles for ' + place.title() + ' image ' + image_suffix + ' using ground-truth \'' + label_suffix + '\' and input stack \'' + stack_label + '\'')
            print('')
            construct_dataset_tiles(data_path, place, tiles, label_stats, image_suffix,
                window, stack_label, feature_count,
                bands_vir=bands_vir,
                bands_sar=bands_sar,
                bands_ndvi=bands_ndvi,
                bands_ndbi=bands_ndbi,
                bands_osm=bands_osm,
                haze_removal=False,
                label_suffix=label_suffix, 
                category_label=category_label )

            print('Combine dataset tiles into complete data arrays')
            X_data, Y_data = combine_dataset_tiles(data_path, place, tiles, label_suffix, image_suffix, stack_label, window, resolution=tile_resolution)

            print('Write complete datasets to file')
            split_dataset(data_path, place, label_suffix, stack_label, image_suffix, window, resolution=tile_resolution)
            print('')

def classify_tile(tile_id, 
        data_path, place, tiles, image_suffix,
        window, stack_label, feature_count, model_id, scaler, model, n_cats,
        bands_vir,
        bands_sar, bands_ndvi, bands_ndbi, bands_osm,
        haze_removal, unflatten_input, water_overwrite, water_mask, water_threshold):
        
    mask, imn, geo, prj = prepare_input_stack(data_path, place, tiles, stack_label, feature_count, 
        image_suffix, window, tile_id, bands_vir=bands_vir, bands_sar=bands_sar, 
        bands_ndvi=bands_ndvi, bands_ndbi=bands_ndbi, bands_osm=bands_osm,
        haze_removal=False)

    Y, Y_deep, Y_max = create_classification_arrays(window, n_cats, imn, tiles['features'][tile_id]['properties']['pad'])

    fill_classification_arrays(feature_count, window, n_cats, scaler, model, imn, Y, Y_deep, Y_max, unflatten_input=unflatten_input, water_overwrite=water_overwrite)
    resolution = int(tiles['features'][0]['properties']['resolution'])


    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))


    result_file = data_path+'maps/'+place+'_tile'+str(tile_id).zfill(zfill)+'_'+model_id+'_lulc_'+image_suffix+'.tif'

    print(result_file)
    util_rasters.write_1band_geotiff(result_file, Y, geo, prj, data_type=gdal.GDT_Byte)
    if np.sum(Y==255) != 0:
        print('unclassified pixels in Y:', np.sum(Y==255))

    Y_deep = np.transpose(Y_deep, (2,0,1))
    
    Y_full = np.zeros((Y_deep.shape[0]+2, Y_deep.shape[1], Y_deep.shape[2]),dtype='float32')
    for b in range(Y_deep.shape[0]):
        Y_full[b] = Y_deep[b]
    b+=1
    Y_full[b][:,:] = Y[:,:]
    b+=1
    Y_full[b][:,:] = Y_max[:,:]
    print('Y_full sample', Y_full[:,100,100])

    full_result_file = data_path+'maps/'+place+'_tile'+str(tile_id).zfill(zfill)+'_'+model_id+'_full_'+image_suffix+'.tif'


    util_rasters.write_multiband_geotiff(full_result_file, Y_full, geo, prj, data_type=gdal.GDT_Float32)

    if water_mask:
        water = util_rasters.make_water_mask_tile(data_path, place, tile_id, tiles, image_suffix, water_threshold)
        water_file = data_path+'maps/'+place+'_tile'+str(tile_id).zfill(zfill)+'_water_'+image_suffix+'.tif'
        print(water_file)
        util_rasters.write_1band_geotiff(water_file, water, geo, prj, data_type=gdal.GDT_Byte)

    del mask, imn, geo, prj, Y, Y_deep, Y_max, Y_full
    print('tile', tile_id, 'done')
    
    return

def map_classify_tile(args):
    try:
        return classify_tile(*args)
    except Exception as e:
        raise e
    return

def classify_tiles(data_path, place, tiles, image_suffix,
        window, stack_label, feature_count, model_id, scaler, model, n_cats,
        bands_vir=['blue','green','red','nir','swir1','swir2'],
        bands_sar=['vv','vh'], bands_ndvi=None, bands_ndbi=None, bands_osm=None,
        haze_removal=False, unflatten_input=False, water_overwrite=False, water_mask=False,
        water_threshold=0.1):
            
    print("Feature count:", feature_count)
    print("Stack label: ", stack_label)
    
    # eg 'vir', 'vir_sar', 'vir_ndvir', 'vir&sar&ndvirnx', 'vir&sar&ndvir', 'vir&sar&ndvirnx&osm', 'vir&ndvirnx&osm', 'vir&dem'
    #for tile_id in [single_tile_id]:
    for tile_id in range(len(tiles['features'])):
        
        classify_tile(tile_id, 
            data_path, place, tiles, image_suffix,
            window, stack_label, feature_count, model_id, scaler, model, n_cats,
            bands_vir,
            bands_sar, bands_ndvi, bands_ndbi, bands_osm,
            haze_removal, unflatten_input, water_overwrite, water_mask, water_threshold)
        

def class_balancing(Y_t, X_train, Y_train):
    # create variables to hold the count of each categories
    n_OpnSp = np.sum(Y_t==0)
    n_NRes = np.sum(Y_t==1)
    n_Res = np.sum(Y_t==4) 
    n_Rd = np.sum(Y_t==6)
        
    print("No of open space = ", n_OpnSp)
    print("No of non residential = ", n_NRes)
    print("No of residential = ", n_Res)
    print("No of roads = ", n_Rd)

    # create a list containing total count of each categories
    tot_cnt = [n_OpnSp, n_NRes, n_Res, n_Rd]

    # get the least representative class
    LRC = min(tot_cnt)
    print("least representative class = ", LRC)

    # use the least representative class to match other, remove extra samples
    #Return a new array of given shape and type, filled with zeros
    Y_balanced = np.zeros((LRC*4),dtype=Y_t.dtype)
    X_balanced = np.zeros((LRC*4,X_train.shape[1]),dtype=X_train.dtype)

    # create four numpy arrays to hold values from each category
    where_OpnSp = np.where(Y_t==0)
    where_NRes = np.where(Y_t==1)
    where_Res = np.where(Y_t==4)
    where_Rd = np.where(Y_t==6) 

    # Remove extra samples using random permutation from each category
    where_OpnSp_array = where_OpnSp[0]
    perm = np.random.permutation(len(where_OpnSp_array))
    where_OpnSp_array = where_OpnSp_array[perm[:]]
    where_OpnSp_trunc = (where_OpnSp_array[:LRC],)

    Y_balanced[:LRC] = Y_t[where_OpnSp_trunc]
    X_balanced[:LRC] = X_train[where_OpnSp_trunc]

    where_NRes_array = where_NRes[0]
    perm = np.random.permutation(len(where_NRes_array))
    where_NRes_array = where_NRes_array[perm[:]]
    where_NRes_trunc = (where_NRes_array[:LRC],)

    Y_balanced[LRC:(LRC*2)] = Y_t[where_NRes_trunc]
    X_balanced[(LRC):(LRC*2)] = X_train[where_NRes_trunc]

    where_Res_array = where_Res[0]
    perm = np.random.permutation(len(where_Res_array))
    where_Res_array = where_Res_array[perm[:]]
    where_Res_trunc = (where_Res_array[:LRC],)

    Y_balanced[(LRC*2):(LRC*3)] = Y_t[where_Res_trunc]
    X_balanced[(LRC*2):(LRC*3)] = X_train[where_Res_trunc]

    where_Rd_array = where_Rd[0]
    perm = np.random.permutation(len(where_Rd_array))
    where_Rd_array = where_Rd_array[perm[:]]
    where_Rd_trunc = (where_Rd_array[:LRC],)

    Y_balanced[(LRC*3):(LRC*4)] = Y_t[where_Rd_trunc]
    X_balanced[(LRC*3):(LRC*4)] = X_train[where_Rd_trunc]

    perm = np.random.permutation(LRC*4)
    Y_balanced = Y_balanced[perm[:]]
    X_balanced = X_balanced[perm[:],:]

    print("Sum of each categories after balancing = ", np.sum(Y_balanced==0),np.sum(Y_balanced==1),np.sum(Y_balanced==4),np.sum(Y_balanced==6))
    print(Y_balanced.shape)

def view_results_tile(data_path, place, tiles, tile_id, model_id, image_suffix,
        category_label={0:'Open Space',1:'Non-Residential',\
                   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
                   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
                   6:'Roads',7:'Study Area',8:'Labeled Study Area',9:'Water',254:'No Data',255:'No Label'} ,
        show_vir=True):
    
    resolution = int(tiles['features'][tile_id]['properties']['resolution'])
    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))

    result_file = data_path+'maps/'+place+'_tile'+str(tile_id).zfill(zfill)+'_'+model_id+'_lulc_'+image_suffix+'.tif'


    util_rasters.stats_byte_raster(result_file, category_label)

    result, geo, prj, cols, rows = util_rasters.load_geotiff(result_file)
    result[result==0] = 0
    result[result==1] = 1
    result[result==2] = 4
    result[result==3] = 6

    rgb = util_imagery.rgb_lulc_result(result)
    fig = plt.figure(figsize=(16,16))
    plt.imshow(rgb)
    print

    if show_vir:
        img_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_vir_'+image_suffix+'.tif'

        print(img_file)
        util_rasters.show_vir_s2(img_file)
    return

def view_results_overlay(data_path, place, tiles, tile_id, model_id, image_suffix,
        category_label={0:'Open Space',1:'Non-Residential',\
                   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
                   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
                   6:'Roads',7:'Study Area',8:'Labeled Study Area',9:'Water',254:'No Data',255:'No Label'} ,
         show_vir=True, show_lulc=True):
    # remapping parameter

    resolution = int(tiles['features'][tile_id]['properties']['resolution'])
    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))
    result_file = data_path+'maps/'+place+'_tile'+str(tile_id).zfill(zfill)+'_'+model_id+'_lulc_'+image_suffix+'.tif'

    util_rasters.stats_byte_raster(result_file, category_label)

    result, geo, prj, cols, rows = util_rasters.load_geotiff(result_file)
    result[result==0] = 0
    result[result==1] = 1
    result[result==2] = 4
    result[result==3] = 6

    rgb = util_imagery.rgb_lulc_result(result)


    img_file = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_vir_'+image_suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.tif'

    img, geo, prj, cols, rows = util_rasters.load_geotiff(img_file)

    img = np.transpose(img, (1,2,0))
    img = img[:,:,0:3]
    img = np.flip(img, axis=-1)

    viz = img.astype('float32')
    viz = viz/3000
    viz = 255.*viz
    viz = np.clip(viz,0,255)
    viz = viz.astype('uint8')
    for b in range(img.shape[2]):
        print(b, np.min(viz[:,:,b]), np.max(viz[:,:,b]))

    plt.figure(figsize=[16,16])
    plt.imshow(viz, interpolation='none')
    plt.imshow(rgb, interpolation='none', alpha=0.3)
    plt.show()

    if show_vir:
        plt.figure(figsize=[16,16])
        plt.imshow(viz, interpolation='none')
        plt.show()
    if show_lulc:
        plt.figure(figsize=[16,16])
        plt.imshow(rgb, interpolation='none')
        plt.show()

    return

def chunk_training_data(data_root, place_images, label_suffix, resolution, stack_label, window, chunk_id_base, 
        training_samples_per_chunk=70000, validation_samples_per_chunk=30000):
    image_names = []
    t_paths = []
    v_paths = []
    t_counts = []
    v_counts = []
    t_perms = []
    v_perms = []
    X_shape = None
    X_dtype = None
    Y_shape = None
    Y_dtype = None

    t_total = 0
    v_total = 0
    for city, suffixes in place_images.items():
        for suffix in suffixes:
            image_names.append("{}_{}".format(city,suffix))
            train_file = data_root+city+'/'+city+'_train_'+label_suffix+'_'+('' if resolution==10 else str(resolution)+'m_')+stack_label+'_'+str(window)+'w_'+suffix+'.pkl'
            print(train_file)
            with open(train_file, 'rb') as f:
                X_train_sub, Y_train_sub = pickle.load(f)
            f.close()
            valid_file = data_root+city+'/'+city+'_valid_'+label_suffix+'_'+('' if resolution==10 else str(resolution)+'m_')+stack_label+'_'+str(window)+'w_'+suffix+'.pkl'
            print(valid_file)
            with open(valid_file, 'rb') as f:
                X_valid_sub, Y_valid_sub = pickle.load(f)
            f.close()
            t_paths.append(train_file)
            v_paths.append(valid_file)
            print(X_train_sub.shape, Y_train_sub.shape, X_valid_sub.shape, Y_valid_sub.shape)
            X_shape = X_train_sub.shape
            X_dtype = X_train_sub.dtype
            Y_shape = Y_train_sub.shape
            Y_dtype = Y_train_sub.dtype
            t_sub = X_train_sub.shape[0]
            v_sub = X_valid_sub.shape[0]
            #print(t_sub, v_sub)
            t_counts.append(t_sub)
            v_counts.append(v_sub)
            t_perms.append(np.random.permutation(t_sub))
            v_perms.append(np.random.permutation(v_sub))
            t_total += t_sub
            v_total += v_sub
    print(t_total, v_total)
    n_images = len(image_names)
    image_shares = []
    for i in range(n_images):
        image_shares.append(float(t_counts[i]) / float(t_total))

    t_samples_per_chunk = training_samples_per_chunk
    v_samples_per_chunk = validation_samples_per_chunk
    n_chunks = (t_total / t_samples_per_chunk) + 1 #plus one for remainder

    t_samples = []
    v_samples = []

    for i in range(n_images):
        
        t_samples.append(int(math.floor((float(t_samples_per_chunk) * image_shares[i]))))
        v_samples.append(int(math.floor((float(v_samples_per_chunk) * image_shares[i]))))

    for chunk_no in range(n_chunks):
        #
        chunk_id = chunk_id_base + '_'+str(chunk_no).zfill(2)+'-'+str(n_chunks-1).zfill(2)

        t_chunk_X = np.zeros(((t_samples_per_chunk,) + X_shape[1:]),dtype=X_dtype)
        t_chunk_Y = np.zeros((t_samples_per_chunk),dtype=Y_dtype)
        v_chunk_X = np.zeros(((v_samples_per_chunk,) + X_shape[1:]),dtype=X_dtype)
        v_chunk_Y = np.zeros((v_samples_per_chunk),dtype=Y_dtype)
        print(t_chunk_X.shape, t_chunk_Y.shape, v_chunk_X.shape, v_chunk_Y.shape)
        
        
        chunk_t_start_idx = 0
        chunk_t_stop_idx = 0
        chunk_v_start_idx = 0
        chunk_v_stop_idx = 0
        for i in range(len(image_names)):
            train_file = t_paths[i]
            print(train_file)
            with open(train_file, 'rb') as f:
                X_train_sub, Y_train_sub = pickle.load(f)
            f.close()
            valid_file = v_paths[i]
            print(valid_file)
            with open(valid_file, 'rb') as f:
                X_valid_sub, Y_valid_sub = pickle.load(f)
            f.close()
            print(X_train_sub.shape, Y_train_sub.shape, X_valid_sub.shape, Y_valid_sub.shape)
            
            if chunk_no == n_chunks-1:
                chunk_t_stop_idx = min(chunk_t_start_idx + t_samples[i], chunk_t_start_idx + t_perms[i][chunk_no*t_samples[i]:(chunk_no+1)*t_samples[i]].shape[0])
                chunk_v_stop_idx = min(chunk_v_start_idx + v_samples[i], chunk_v_start_idx + v_perms[i][chunk_no*v_samples[i]:(chunk_no+1)*v_samples[i]].shape[0])
            else: 
                chunk_t_stop_idx = (chunk_t_start_idx + t_samples[i])
                chunk_v_stop_idx = (chunk_v_start_idx + v_samples[i])
                
            print('FROM ('+image_names[i]+')', chunk_no*t_samples[i], ':', (chunk_no+1)*t_samples[i], 'INTO (chunk_'+str(chunk_no)+')', chunk_t_start_idx, ':', chunk_t_stop_idx)
            t_chunk_X[chunk_t_start_idx:chunk_t_stop_idx] = X_train_sub[t_perms[i][chunk_no*t_samples[i]:(chunk_no+1)*t_samples[i]]]
            t_chunk_Y[chunk_t_start_idx:chunk_t_stop_idx] = Y_train_sub[t_perms[i][chunk_no*t_samples[i]:(chunk_no+1)*t_samples[i]]]
            v_chunk_X[chunk_v_start_idx:chunk_v_stop_idx] = X_train_sub[v_perms[i][chunk_no*v_samples[i]:(chunk_no+1)*v_samples[i]]]
            v_chunk_Y[chunk_v_start_idx:chunk_v_stop_idx] = Y_train_sub[v_perms[i][chunk_no*v_samples[i]:(chunk_no+1)*v_samples[i]]]

            chunk_t_start_idx = chunk_t_stop_idx
            chunk_v_start_idx = chunk_v_stop_idx
            
        if True or chunk_no == n_chunks-1:
            #print('empty rows (training):', np.sum(np.all(t_chunk_X==0, axis=1)))
            nonempty_rows = ~np.all(t_chunk_X==0, axis=1)
            t_chunk_X = t_chunk_X[nonempty_rows]
            t_chunk_Y = t_chunk_Y[nonempty_rows]
            nonempty_rows = ~np.all(v_chunk_X==0, axis=1)
            v_chunk_X = v_chunk_X[nonempty_rows]
            v_chunk_Y = v_chunk_Y[nonempty_rows]
            
        #now write chunk
        chunk_file = data_root+'chunks/'+chunk_id+'_'+label_suffix+'_'+stack_label+'_'+str(window)+'w.pkl'
        print('Writing chunk:', chunk_file)
        # print(np.sum(np.all(t_chunk_X==0,axis=1)))
        pickle.dump((t_chunk_X, t_chunk_Y, v_chunk_X, v_chunk_Y), open(chunk_file, 'wb'))


def water_mask_tiles(data_path, place, tiles, image_suffix, water_threshold=0.15):
    resolution = int(tiles['features'][0]['properties']['resolution'])
    if resolution==10:
        zfill = 3
    elif resolution==5:
        zfill = 4
    else:
        raise Exception('bad tile resolution: '+str(tiles['features'][0]['properties']['resolution']))
    for tile_id in len(tiles['features']):
        water = util_rasters.make_water_mask_tile(data_path, place, tile_id, tiles, image_suffix, water_threshold)
        water_file = data_path+'maps/'+place+'_tile'+str(tile_id).zfill(zfill)+'_water_'+image_suffix+'.tif'
        print(water_file)
        util_rasters.write_1band_geotiff(water_file, water, geo, prj, data_type=gdal.GDT_Byte)


def apply_model_to_data_3category(
        model_id, notes, place_images,
        data_root, resolution, window, 
        bands_vir=['blue','green','red','nir','swir1','swir2'],
        bands_sar=None, bands_ndvi=None, bands_ndbi=None, bands_osm=None,
        label_suffix='aue',
        unflatten_input=True):
    if resolution==10:
        zfill = 3
    elif resolution==5:
        zfill = 4
    else:
        raise Exception('bad tile_resolution: '+str(tile_resolution))

    #load scaler & network
    scaler_filename = data_root+'models/'+model_id+'_scaler.pkl'
    network_filename = data_root+'models/'+model_id+'.hd5'
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)
    f.close()
    network = load_model(network_filename, custom_objects={'loss': 'categorical_crossentropy'})

    stack_label, feature_count = build_stack_label(
            bands_vir=bands_vir,
            bands_sar=bands_sar,
            bands_ndvi=bands_ndvi,
            bands_ndbi=bands_ndbi,
            bands_osm=bands_osm,)

    cats_map = {}
    cats_map[0] = 0
    cats_map[1] = 1
    cats_map[2] = 4
    cats_map[3] = 4
    cats_map[4] = 4
    cats_map[5] = 4
    cats_map[6] = 6

    categories = [0,1,4,6]

    for city, image_suffixes in place_images.items():
        for suffix in image_suffixes:
            data_path = data_root + city + '/'

            train_file = data_root+city+'/'+city+'_train_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(train_file)
            with open(train_file, 'rb') as f:
                X_train_raw, Y_train_raw = pickle.load(f)
            f.close()
            valid_file = data_root+city+'/'+city+'_valid_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(valid_file)
            with open(valid_file, 'rb') as f:
                X_valid_raw, Y_valid_raw = pickle.load(f)
            f.close()
            print(X_train_raw.shape, Y_train_raw.shape, X_valid_raw.shape, Y_valid_raw.shape)

            X_train_raw_scaled, X_valid_raw_scaled, scaler = util_ml.scale_learning_data(X_train_raw, X_valid_raw)
            print(X_train_raw_scaled.shape,  X_valid_raw_scaled.shape)
            del X_train_raw, X_valid_raw

            if(unflatten_input):
                X_train = X_train_raw_scaled.reshape((X_train_raw_scaled.shape[0],feature_count,window,window))
                X_valid = X_valid_raw_scaled.reshape((X_valid_raw_scaled.shape[0],feature_count,window,window))
            else:
                X_train = X_train_raw_scaled
                X_valid = X_valid_raw_scaled

            del X_train_raw_scaled, X_valid_raw_scaled
            print(X_train.shape)

            Y_train = Y_train_raw.copy()
            Y_valid = Y_valid_raw.copy()

            for k, v in cats_map.items():
                Y_train[Y_train_raw==k] = v
                Y_valid[Y_valid_raw==k] = v
                
            print(Y_train_raw.shape)
            print(Y_train.shape)

            del Y_train_raw, Y_valid_raw

            non_roads = np.where(Y_train!=6)
            Y_train = Y_train[non_roads]
            X_train = X_train[non_roads]
            non_roads = np.where(Y_valid!=6)
            Y_valid = Y_valid[non_roads]
            X_valid = X_valid[non_roads]

            Y_t = Y_train.copy()
            Y_v = Y_valid.copy()

            Y_t[Y_train==0] = 0
            Y_t[Y_train==1] = 1
            Y_t[Y_train==4] = 2
            Y_t[Y_train==6] = 3

            Y_v[Y_valid==0] = 0
            Y_v[Y_valid==1] = 1
            Y_v[Y_valid==4] = 2
            Y_v[Y_valid==6] = 3

            categories_reduced = [0,1,2]

            print("evaluate training")
            Yhat_t_prob = network.predict(X_train)
            Yhat_t = Yhat_t_prob.argmax(axis=-1)
            train_confusion = util_ml.calc_confusion(Yhat_t,Y_t,categories_reduced)
            train_recalls, train_precisions, train_accuracy = util_ml.calc_confusion_details(train_confusion)

            print("evaluate validation")
            Yhat_v_prob = network.predict(X_valid)
            Yhat_v = Yhat_v_prob.argmax(axis=-1)
            valid_confusion = util_ml.calc_confusion(Yhat_v,Y_v,categories_reduced)
            valid_recalls, valid_precisions, valid_accuracy = util_ml.calc_confusion_details(valid_confusion)


            # Calculate f-score
            beta = 2
            train_f_score = (beta**2 + 1) * train_precisions * train_recalls / ( (beta**2 * train_precisions) + train_recalls )
            train_f_score_open = train_f_score[0] 
            train_f_score_nonres = train_f_score[1]  
            train_f_score_res = train_f_score[2]  
            train_f_score_roads = None#train_f_score[3]  
            train_f_score_average = np.mean(train_f_score)

            # Calculate f-score
            valid_f_score = (beta**2 + 1) * valid_precisions * valid_recalls / ( (beta**2 * valid_precisions) + valid_recalls )
            valid_f_score_open = valid_f_score[0] 
            valid_f_score_nonres = valid_f_score[1] 
            valid_f_score_res = valid_f_score[2] 
            valid_f_score_roads = None# valid_f_score[3] 
            valid_f_score_average = np.mean(valid_f_score)

            # expanding lists to match expected model_record stuff
            train_recalls_expanded = [train_recalls[0],train_recalls[1],train_recalls[2],None]
            valid_recalls_expanded = [valid_recalls[0],valid_recalls[1],valid_recalls[2],None]
            train_precisions_expanded = [train_precisions[0],train_precisions[1],train_precisions[2],None]
            valid_precisions_expanded = [valid_precisions[0],valid_precisions[1],valid_precisions[2],None]

            util_workflow.record_model_application(
                    model_id, notes, place_images, label_suffix, resolution, stack_label, feature_count, window, cats_map, 
                    train_confusion, train_recalls_expanded, train_precisions_expanded, train_accuracy,
                    train_f_score_open, train_f_score_nonres, train_f_score_res, train_f_score_roads, train_f_score_average,
                    valid_confusion, valid_recalls_expanded, valid_precisions_expanded, valid_accuracy,
                    valid_f_score_open, valid_f_score_nonres, valid_f_score_res, valid_f_score_roads, valid_f_score_average,)


def apply_model_to_data_1vAll(
        model_id, notes, place_images,
        data_root, resolution, window, 
        bands_vir=['blue','green','red','nir','swir1','swir2'],
        bands_sar=None, bands_ndvi=None, bands_ndbi=None, bands_osm=None,
        label_suffix='aue',
        unflatten_input=True):
    if resolution==10:
        zfill = 3
    elif resolution==5:
        zfill = 4
    else:
        raise Exception('bad tile_resolution: '+str(tile_resolution))

    #load scaler & network
    scaler_filename = data_root+'models/'+model_id+'_scaler.pkl'
    network_filename = data_root+'models/'+model_id+'.hd5'
    with open(scaler_filename, "rb") as f:
        scaler = pickle.load(f)
    f.close()
    network = load_model(network_filename, custom_objects={'loss': 'categorical_crossentropy'})

    stack_label, feature_count = build_stack_label(
            bands_vir=bands_vir,
            bands_sar=bands_sar,
            bands_ndvi=bands_ndvi,
            bands_ndbi=bands_ndbi,
            bands_osm=bands_osm,)

    cats_map = {}
    cats_map[0] = 0
    cats_map[1] = 1
    cats_map[2] = 4
    cats_map[3] = 4
    cats_map[4] = 4
    cats_map[5] = 4
    cats_map[6] = 6

    categories = [0,1,4,6]

    for city, image_suffixes in place_images.items():
        for suffix in image_suffixes:
            data_path = data_root + city + '/'

            train_file = data_root+city+'/'+city+'_train_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(train_file)
            with open(train_file, 'rb') as f:
                X_train_raw, Y_train_raw = pickle.load(f)
            f.close()
            valid_file = data_root+city+'/'+city+'_valid_'+label_suffix+'_'+stack_label+'_'+str(window)+'w_'+suffix+('' if resolution==10 else '_'+str(resolution)+'m')+'.pkl'
            print(valid_file)
            with open(valid_file, 'rb') as f:
                X_valid_raw, Y_valid_raw = pickle.load(f)
            f.close()
            print(X_train_raw.shape, Y_train_raw.shape, X_valid_raw.shape, Y_valid_raw.shape)

            X_train_raw_scaled, X_valid_raw_scaled, scaler = util_ml.scale_learning_data(X_train_raw, X_valid_raw)
            print(X_train_raw_scaled.shape,  X_valid_raw_scaled.shape)
            del X_train_raw, X_valid_raw

            if(unflatten_input):
                X_train = X_train_raw_scaled.reshape((X_train_raw_scaled.shape[0],feature_count,window,window))
                X_valid = X_valid_raw_scaled.reshape((X_valid_raw_scaled.shape[0],feature_count,window,window))
            else:
                X_train = X_train_raw_scaled
                X_valid = X_valid_raw_scaled

            del X_train_raw_scaled, X_valid_raw_scaled
            print(X_train.shape)

            Y_train = Y_train_raw.copy()
            Y_valid = Y_valid_raw.copy()

            for k, v in cats_map.items():
                Y_train[Y_train_raw==k] = v
                Y_valid[Y_valid_raw==k] = v

            NB_FILTERS_1=32
            NB_FILTERS_2=64

            cats=[0,1]
            networks = []
            # for cat in [6,4,1,0]:
            cat = 6
                        
            # create binary data
            Y_t_bin = util_ml.make_binary(Y_train, cat, silent=True)
            Y_v_bin = util_ml.make_binary(Y_valid, cat, silent=True)
                

            # score model
            print("evaluate training")
            Yhat_t_prob = network.predict(X_train)
            Yhat_t = np.rint(Yhat_t_prob).astype(int).reshape(Yhat_t_prob.shape[0])
            # print(Yhat_t_prob[0:20])
            # print(Yhat_t[0:20])
            #Yhat_t = Yhat_t_prob.argmax(axis=-1)
            train_confusion = util_ml.calc_confusion(Yhat_t,Y_t_bin,cats)
            train_recalls, train_precisions, train_accuracy = util_ml.calc_confusion_details(train_confusion)
            print(train_recalls, train_precisions)
            print()

            print("evaluate validation")
            Yhat_v_prob = network.predict(X_valid)
            Yhat_v = np.rint(Yhat_v_prob).astype(int).reshape(Yhat_v_prob.shape[0])
            # print(Yhat_v_prob[0:20])
            # print(Yhat_v[0:20])
            #Yhat_v = Yhat_v_prob.argmax(axis=-1)
            valid_confusion = util_ml.calc_confusion(Yhat_v,Y_v_bin,cats)
            valid_recalls, valid_precisions, valid_accuracy = util_ml.calc_confusion_details(valid_confusion)
            print(valid_recalls, valid_precisions)
            print()
                
            # Calculate f-score
            beta = 2
            train_f_score = (beta**2 + 1) * train_precisions * train_recalls / ( (beta**2 * train_precisions) + train_recalls )
            train_f_score_open = train_f_score[0] 
            train_f_score_nonres = None#train_f_score[1]  
            train_f_score_res = None#train_f_score[2]  
            train_f_score_roads = train_f_score[1]  
            train_f_score_average = np.mean(train_f_score)
            # Calculate f-score
            valid_f_score = (beta**2 + 1) * valid_precisions * valid_recalls / ( (beta**2 * valid_precisions) + valid_recalls )
            valid_f_score_open = valid_f_score[0] 
            valid_f_score_nonres = None#valid_f_score[1] 
            valid_f_score_res = None#valid_f_score[2] 
            valid_f_score_roads = valid_f_score[1] 
            valid_f_score_average = np.mean(valid_f_score)
            # expanding lists to match expected model_record stuff
            train_recalls_expanded = [train_recalls[0],None,None,train_recalls[1]]
            valid_recalls_expanded = [valid_recalls[0],None,None,valid_recalls[1]]
            train_precisions_expanded = [train_precisions[0],None,None,train_precisions[1]]
            valid_precisions_expanded = [valid_precisions[0],None,None,valid_precisions[1]]
            util_workflow.record_model_application(
                    model_id, notes, place_images, label_suffix, resolution, stack_label, feature_count, window, cats_map, 
                    train_confusion, train_recalls_expanded, train_precisions_expanded, train_accuracy,
                    train_f_score_open, train_f_score_nonres, train_f_score_res, train_f_score_roads, train_f_score_average,
                    valid_confusion, valid_recalls_expanded, valid_precisions_expanded, valid_accuracy,
                    valid_f_score_open, valid_f_score_nonres, valid_f_score_res, valid_f_score_roads, valid_f_score_average,)

