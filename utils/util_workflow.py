# organized set of helper functions
# drawn from bronco.py and bronco notebooks
# topic: building training data

import warnings
warnings.filterwarnings('ignore')
#
#import os
#import sys
#import json
#import itertools
#import pickle
#from pprint import pprint
#
import numpy as np
#import shapely
#import cartopy
from osgeo import gdal
#import matplotlib.pyplot as plt
#
#import descarteslabs as dl




def build_stack_label(bands_vir, bands_sar, bands_ndvi_raw, bands_ndvi_min, bands_ndvi_max, 
                      bands_ndbi_raw, bands_ndbi_min, bands_ndbi_max, bands_osm_roads,
                     pair_vir=False, pair_sar=False, pair_ndvi_raw=False, pair_ndbi_raw=False):
    feature_count = 0
    stack_label = ''
    if bands_vir is not None:
        if pair_vir:
            feature_count += 12
            stack_label += '2vir+'
        else:
            feature_count += 6
            stack_label += 'vir+'
    if bands_sar is not None:
        if pair_sar:
            feature_count += 4
            stack_label += '2sar+'
        else:
            feature_count += len(bands_sar)
            stack_label += 'sar+'
    if bands_ndvi_raw is not None:
        if pair_ndvi_raw:
            feature_count += 2
            stack_label += '2ndvir+'
        else:
            feature_count += 1
            stack_label += 'ndvir+'
    if bands_ndvi_min is not None:
        feature_count += 1
        stack_label += 'ndvin+'
    if bands_ndvi_max is not None:
        feature_count += 1
        stack_label += 'ndvix+'
    if bands_ndbi_raw is not None:
        if pair_ndbi_raw:
            feature_count += 2
            stack_label += '2ndbir+'
        else:
            feature_count += 1
            stack_label += 'ndbir+'
    if bands_ndbi_min is not None:
        feature_count += 1
        stack_label += 'ndbin+'
    if bands_ndbi_max is not None:
        feature_count += 1
        stack_label += 'ndbix+'
    if bands_osm_roads is not None:
        feature_count += 1
        stack_label += 'osm+'
    if stack_label.endswith('+'):
        stack_label = stack_label[:-1]
    return stack_label, feature_count

def construct_dataset_tiles(data_path, place, tiles, label_stats, image_suffix,
        bands_vir, bands_sar, bands_ndvi_raw, bands_ndvi_min, bands_ndvi_max, bands_osm_roads,
        d, stack_label, feature_count, haze_removal=False,
        label_suffix='labels', categories=[0,1,4,6], 
        category_label={0:'Open Space',1:'Non-Residential',\
                   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
                   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
                   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'} ):
    
    r = d/2

    print "Feature count:", feature_count
    print "Stack label: ", stack_label
    # eg 'vir', 'vir_sar', 'vir_ndvir', 'vir&sar&ndvirnx', 'vir&sar&ndvir', 'vir&sar&ndvirnx&osm', 'vir&ndvirnx&osm', 'vir&dem'
    # for tile_id in [19]:
    for tile_id in range(len(tiles['features'])):
        # skip tiles without labels
        # HARDCODED THRESHOLD
        if (len(label_stats[tile_id].keys())==1) or (label_stats[tile_id][255]<40000):
            # print 'WARNING: tile', tile_id, ' has no labels'
            continue
        tile = tiles['features'][tile_id]
        side_length = tile['properties']['tilesize'] + tile['properties']['pad']*2

        imn = np.zeros((feature_count,side_length,side_length),dtype='float32')
        n_features = imn.shape[0] 

        print 'tile', tile_id, 'load VIR image'
        vir_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_vir_'+image_suffix+'.tif'
        vir, virgeo, virprj, vircols, virrows = bronco.load_geotiff(vir_file,dtype='uint16')
        print 'vir shape:',vir.shape
        vir = vir.astype('float32')
        vir = vir/10000.
        vir = np.clip(vir,0.0,1.0)
        #print 'tile', tile_id, 'make data mask from vir alpha'
        mask = (vir[6][:,:] > 0)  # vir[6] is the alpha band in the image, takes values 0 and 65535
        nodata = (vir[6][:,:]==0)
        print np.sum(mask), "study area within image"
        print mask.shape[0] * mask.shape[1], "full extent of image"
        # haze removal
        if(haze_removal):
            virc, virc_ro = bronco.ls_haze_removal(vir[:-1],nodata)
            print virc_ro
            vir[:-1] = virc[:]

        b_start = 0
        if bands_vir is not None:
            for b in range(vir.shape[0]-1):
                print 'vir band',b,'into imn band',b_start+b,'(',np.min(vir[b,:,:]),'-',np.max(vir[b,:,:]),')'
                imn[b_start+b][:,:] = vir[b][:,:]
            b_start += vir.shape[0]-1

        if bands_sar is not None:
            print 'tile', tile_id, 'load SAR image'
            sar_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_sar_'+image_suffix+'.tif'
            sar, sargeo, sarprj, sarcols, sarrows = bronco.load_geotiff(sar_file,dtype='uint16')
            print 'sar shape:',sar.shape
            sar = sar.astype('float32')
            sar = sar/255.
            sar = np.clip(sar,0.0,1.0)
            for b in range(sar.shape[0]):
                print 'sar band',b,'into imn band',b_start+b,'(',np.min(sar[b,:,:]),'-',np.max(sar[b,:,:]),')'
                imn[b_start+b][:,:] = sar[b][:,:]
            b_start += sar.shape[0]

        if bands_ndvi_raw is not None:
            #print 'tile', tile_id, 'calculate NDVI raw'
            # returns (a - b)/(a + b)
            tol=1e-6
            a_minus_b = np.add(vir[3,:,:],np.multiply(vir[2,:,:],-1.0))
            a_plus_b = np.add(np.add(vir[3,:,:],vir[2,:,:]),tol)
            y = np.divide(a_minus_b,a_plus_b)
            y = np.clip(y,-1.0,1.0)
            a_minus_b = None
            a_plus_b = None
            ndvi_raw = y #nir, red
            print 'ndvi_raw shape:', ndvi_raw.shape
            print 'ndvi raw into imn band',b_start,'(',np.min(ndvi_raw),'-',np.max(ndvi_raw),')'
            imn[b_start] = ndvi_raw
            b_start += 1   

        if bands_ndvi_min is not None:
            print 'tile', tile_id, 'load NDVI min'
            ndvi_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_ndvimin.tif'
            ndvimin, ndvigeo, ndviprj, ndvicols, ndvirows = bronco.load_geotiff(ndvi_file,dtype='float32')
            if(np.sum(np.isnan(ndvimin)) > 0):
                ndvi_nan = np.isnan(ndvimin)
                print 'nan ndvi inside study area:',np.sum(np.logical_and(ndvi_nan, mask))
                ndvimin[ndvi_nan]=0
            print 'ndvi min into imn band',b_start,'(',np.min(ndvimin),'-',np.max(ndvimin),')'
            imn[b_start] = ndvimin
            b_start += 1

        if bands_ndvi_max is not None:
            print 'tile', tile_id, 'load NDVI max'
            ndvi_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_ndvimax.tif'
            ndvimax, ndvigeo, ndviprj, ndvicols, ndvirows = bronco.load_geotiff(ndvi_file,dtype='float32')
            if(np.sum(np.isnan(ndvimax)) > 0):
                ndvi_nan = np.isnan(ndvimax)
                print 'nan ndvi inside study area:',np.sum(np.logical_and(ndvi_nan, mask))
                ndvimax[ndvi_nan]=0
            print 'ndvi max into imn band',b_start,'(',np.min(ndvimax),'-',np.max(ndvimax),')'
            imn[b_start] = ndvimax
            b_start += 1

        if bands_osm_roads is not None:
            print 'tile', tile_id, 'load OSM roads'
            osm_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_osm.tif'
            osm, osmgeo, osmprj, osmcols, osmrows = bronco.load_geotiff(osm_file,dtype='uint8')
            osm[osm==255] = 0
            osm = osm.astype('float32')
            osm = np.clip(osm,0.0,1.0)
            print 'osm roads into imn band',b_start,'(',np.min(osm),'-',np.max(osm),')'
            imn[b_start] = osm
            b_start += 1

        print 'imn', imn.shape, n_features
        print 'tile', tile_id, 'load labels'
        label_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_'+label_suffix+'.tif'
        print label_file
        lb, lbgeo, lbprj, lbcols, lbrows = bronco.load_geotiff(label_file,dtype='uint8')
        #print "NYU AoUE labels", label_file, lbcols, lbrows, lbgeo, lbprj
        # delete training points close to edge
        lb[0:r,:] = 255; lb[-r-1:,:] = 255
        lb[:,0:r] = 255; lb[:,-r-1:] = 255
        y = np.zeros((9,mask.shape[0],mask.shape[1]),dtype='byte')
        y[0] = (mask==1); y[0] &= (lb==0)
        y[1] = (lb==1)
        y[2] = (lb==2)
        y[3] = (lb==3)#; y[3] |= (lb==2)  # merge categories 2 and 3
        #y[4] = (lb==4); y[4] |= (lb==5)  # merge categories 4 and 5
        #change for 4-category typology that consolidates all residential types
        y[4] = (lb==4); y[4] |= (lb==5); y[4] |= (lb==2); y[4] |= (lb==3) # merge categories 2,3,4,5
        y[5] = (lb==5)
        y[6] = (lb==6)
        y[7] = (mask==1)
        y[8] = (lb!=255)
        print 'y.shape', y.shape
        for i in range(9):
            print i, np.sum(y[i]), category_label[i] 
        print 'tile', tile_id, 'collect data,label samples'
        ## unbalanced training
        n_samples = {}
        n_all_samples = 0
        for c in categories:
            n_samples[c] = np.sum(y[c])
            n_all_samples = n_all_samples + n_samples[c]
        print "n_samples, sum", n_samples, n_all_samples
        X_data = np.zeros((n_all_samples,d*d*n_features),dtype=imn.dtype)  # imn2
        Y_data = np.zeros((n_all_samples),dtype='uint8')
        print "X,Y shapes", X_data.shape, Y_data.shape
        index = 0
        for ki in range(len(categories)):
            try:
                k = categories[ki]
                n_k = np.sum((y[k] == 1))
                if (n_k==0):
                    print 'WARNING: tile', tile_id, 'category', ki, n_k, 'no training examples, continuing'
                    continue
                print k, categories[ki], n_k
                z_k = np.where((y[k]==1))
                n_k = len(z_k[0])
                if n_k != n_samples[k]:
                    print "error! mismatch",n_k, n_samples[k] 
                X_k = np.zeros((d*d*n_features,n_k),imn.dtype)  # imn2
                for s in range(n_k):
                    w = bronco.window(imn,z_k[0][s],z_k[1][s],r) # imn2
                    X_k[:,s] = w.flatten()
                X_k = X_k.T

                X_k_nan = np.isnan(X_k)
                if(np.sum(X_k_nan) > 0):
                    print 'NaN in training data'
                    print np.where(X_k_nan)
                #perm = np.random.permutation(X_k.shape[0])
                #X_k = X_k[perm[:],:]
                Y_k = np.full((n_samples[k]), fill_value=k, dtype='uint8')
                X_data[index:index+n_samples[k],:] = X_k[:,:]
                Y_data[index:index+n_samples[k]] = Y_k[:]
                index = index + n_samples[k]
                print k, index, X_k.shape, Y_k.shape
            except:
                print 'ERROR: tile', tile_id, 'category', ki,'error, continuing'
        print X_data.shape, Y_data.shape, X_data.dtype
        if ((n_all_samples > 0) and (np.sum((y[0] == 1)) < 30000)):  # <<<< WARNING: HARD-WIRED LIMIT
            label_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_'+label_suffix+'_'+stack_label+'_'+image_suffix+'.pkl'
            print label_file
            pickle.dump((X_data,Y_data), open(label_file, 'wb'))
        else:
            print 'n_all_samples:', n_all_samples, 'mask true:', np.sum((y[0]==1))
            print 'WARNING: tile', tile_id, ' defective tile', n_all_samples, np.sum((y[0] == 1)) 
        # del imn, X_data, Y_data
        print 'tile', tile_id, 'done'
        print '' #line between tiles in output for readability

def combine_dataset_tiles(data_path, place, tiles, label_suffix, image_suffix, stack_label):
    n_samples = 0
    n_features = 0
    n_dtype = 'none'
    # for tile_id in [single_tile_id]:
    for tile_id in range(len(tiles['features'])):
        label_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_'+label_suffix+'_'+stack_label+'_'+image_suffix+'.pkl'
        #print label_file

        try:
            with open(label_file, "rb") as f:
                Xt, Yt = pickle.load(f)
            f.close()
        except:
            #print 'tile', str(tile_id).zfill(3), 'has no training samples'
            continue

        print tile_id, Xt.shape, Yt.shape
        n_samples = n_samples + Yt.shape[0]
        if (n_features==0):
            n_features = Xt.shape[1]
            n_dtype = Xt.dtype
        assert n_features==Xt.shape[1]
        assert n_dtype==Xt.dtype

    print n_samples, n_features, n_dtype

    X_data = np.zeros((n_samples,n_features),dtype=n_dtype)
    Y_data = np.zeros((n_samples),dtype='uint8')

    print X_data.shape, Y_data.shape

    n_start = 0
    # for tile_id in [single_tile_id]:
    for tile_id in range(len(tiles['features'])):
        label_file = data_path+place+'_tile'+str(tile_id).zfill(3)+'_'+label_suffix+'_'+stack_label+'_'+image_suffix+'.pkl'
        # print label_file

        try:
            with open(label_file, "rb") as f:
                Xt, Yt = pickle.load(f)
            f.close()
        except:
            #print 'tile', str(tile_id).zfill(3), 'has no training samples'
            continue

        # print tile_id, Xt.shape, Yt.shape
        n_t = Yt.shape[0]
        n_end = n_start + n_t
        X_data[n_start:n_end,:] = Xt[:,:]
        Y_data[n_start:n_end] = Yt[:]
        print n_start, n_end
        n_start = n_end
    print X_data.shape, Y_data.shape
    return X_data, Y_data

def split_dataset(data_path, place, label_suffix, stack_label, image_suffix):
    data_file = data_path+place+'_data_'+label_suffix+'_'+stack_label+'_'+image_suffix+'.pkl'
    print data_file
    with open(data_file, "rb") as f:
        X_data, Y_data = pickle.load(f)
    f.close()
    
    n_samples = Y_data.shape[0]
    
    perm_file = data_path+place+'_perm_'+label_suffix+'.pkl'
    print perm_file
    try:
        with open(perm_file, "rb") as f:
            perm = pickle.load(f)
    except IOError as e:
        print 'Unable to open file:', perm_file #Does not exist OR no read permissions
        print 'Create permutation of length', str(n_samples)
        perm = np.random.permutation(n_samples)
        pickle.dump((perm), open(perm_file, 'wb'))
        
    print len(perm), perm

    if Y_data.shape[0] != perm.shape[0]:
        print 'Cannot use indicated permutation to generate training and validation files from data file:', data_file
        print 'permutation has shape', perm.shape
        print 'X_data has shape', X_data.shape
        return

        
    X_data = X_data[perm[:],:]
    Y_data = Y_data[perm[:]]

    data_scale = 1.
    X_data = X_data/data_scale

    n_train = int(math.floor(0.70*n_samples))
    n_valid = n_samples - n_train
    print n_samples, n_train, n_valid

    X_train = X_data[:n_train,:]
    X_valid = X_data[n_train:,:]

    Y_train = Y_data[:n_train]
    Y_valid = Y_data[n_train:]

    print X_train.shape, Y_train.shape
    print X_valid.shape, Y_valid.shape

    train_file = data_path+place+'_train_'+label_suffix+'_'+stack_label+'_'+image_suffix+'.pkl'
    pickle.dump((X_train,Y_train), open(train_file, 'wb'))
    valid_file = data_path+place+'_valid_'+label_suffix+'_'+stack_label+'_'+image_suffix+'.pkl'
    pickle.dump((X_valid,Y_valid), open(valid_file, 'wb'))