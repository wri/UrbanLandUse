# copied directly from bronco repo

import warnings
warnings.filterwarnings('ignore')
#
import os
import sys

import fiona
import subprocess
from datetime import datetime, timedelta
import numpy as np
import shapely
import matplotlib.pyplot as plt

sys.path.append('/home/Peter.Kerins/bronco')
import bronco
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

import gdal

#import json
import itertools
import pickle
from pprint import pprint

import math

#
#import cartopy
#from osgeo import gdal
#
import descarteslabs as dl

def info_studyareas(data_path, place):
    # NYU AoUE Study Region shape
    #print place, place.title() # capitalized version of place name
    place_title = place.title()

    place_shapefile = data_path+place_title+"_studyArea.shp"
    command = 'ogrinfo -al -so {0}'.format(place_shapefile)   # the command goes here
    print '>>>',command
    print subprocess.check_output(command.split(), shell=False)
    
    #!ogrinfo -al -so $ZINSHPNAME
    place_shapefile = data_path+place_title+"_studyAreaEPSG4326.shp"
    command = 'ogrinfo -al -so {0}'.format(place_shapefile)   # the command goes here
    print '>>>',command
    print subprocess.check_output(command.split(), shell=False)
    # !ogr2ogr -t_srs EPSG:4326 $ZOUTSHPNAME $ZINSHPNAME  # First time only
    return

def load_shape(place_shapefile):
    c = fiona.open(place_shapefile)
    pol = c.next()
    shape = {}
    shape['type'] = pol['type']
    shape['properties'] = pol['properties']
    shape['geometry'] = {}
    shape['geometry']['type'] = 'Polygon'  # pol['geometry']['type']
    shape['geometry']['coordinates'] = [[]]
    # if MultiPolygon (e.g., city='kampala')
    if (len(pol['geometry']['coordinates'])>1):
        # identify largest single polygon
        print "MultiPolygon", len(pol['geometry']['coordinates'])
        p_argmax = 0 
        pn_max = 0
        for p in range(len(pol['geometry']['coordinates'])):
            pn = len(pol['geometry']['coordinates'][p][0])
            if pn>pn_max:
                p_argmax = p
                pn_max = pn
            print p, pn, p_argmax, pn_max 
        # make largest polygon the only polygon, move other polys to a backup variable 
        polygon = pol['geometry']['coordinates'][p_argmax]
    else:
        print 'simple polygon'
        polygon = pol['geometry']['coordinates']
       
    xmin =  180
    xmax = -180
    ymin =  90
    ymax = -90
    for x,y in polygon[0]:
        xmin = xmin if xmin < x else x
        xmax = xmax if xmax > x else x
        ymin = ymin if ymin < y else y
        ymax = ymax if ymax > y else y
        shape['geometry']['coordinates'][0].append([x,y])
    shape['bbox'] = [xmin,ymin,xmax,ymax]
    
    return shape

def info_dl_products(silent=False):
    # satellite constellations available in the Descartes API
    sources = dl.metadata.sources()
    # pprint(sources)
    groups = {}
    for group, items in itertools.groupby(sources, key=lambda v: v['product']):
        groups[group] = list(items)

    if(not silent):
        pprint(groups.keys())
    return groups

def info_dl_bands(product='sentinel-2:L1C', silent=False):
    bands_info = dl.metadata.bands(product)
    if(not silent):
        print bands_info[0].keys()
        # import operator
        # bands_info.sort(key=operator.itemgetter('name'))
        for b in range(len(bands_info)):
            band_info = bands_info[b]
            print b, band_info['name'], band_info['description'],
            try: 
                print band_info['name_vendor'], band_info['vendor_order']
            except:
                print 'derived band'
                continue
    return bands_info

def info_dl_vendor_bands(silent=False):
    vendor_bands_info = {}
    bands_info = info_dl_bands(silent=True)
    for b in range(len(bands_info)):
        band_info = bands_info[b]
        try:
            if band_info['name_vendor']!='':
                vendor_bands_info[band_info['name']] = bands_info[b]
                if(not silent): print b, band_info['name'], 'add'
        except:
            if(not silent): print b, band_info['name'], 'skip'
            continue
    print 
    bands = [ u'coastal-aerosol', u'blue', u'green', u'red',  u'nir',  u'swir1', u'swir2'] # S2
    for key in bands:
        try:
            if(not silent): print vendor_bands_info[key]['name'], vendor_bands_info[key]['name_vendor'],
            if(not silent): print 'physical_range', vendor_bands_info[key]['physical_range'],
            if(not silent): print 'valid_range', vendor_bands_info[key]['data_range'],
            if(not silent): print 'nbits', vendor_bands_info[key]['nbits'],
            if(not silent): print 'resolution', vendor_bands_info[key]['resolution'],
        except:
            if(not silent): print 'absent for this satellite',
        if(not silent): print ' '
    return vendor_bands_info

def get_lookback_ids(ids, shape, start_time=None, end_time=None, days=365, satellites=['S2A','S2B'], cloud_fraction_0=0.2, limit=200):
    if start_time==None or end_time==None:
        acquireds = [dl.metadata.get(id)['acquired'] for id in ids]
        acquireds.sort()
        latest_stamp = acquireds[len(acquireds)-1]
        end_time = latest_stamp[0:10]
        end_dt = datetime.strptime(end_time, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=days)
        start_time = start_dt.strftime('%Y-%m-%d')
        print start_time, end_time

    feature_collection_lookback = dl.metadata.search(sat_id=satellites, start_time=start_time, end_time=end_time, 
                                            cloud_fraction_0=cloud_fraction_0, limit=limit, geom=shape['geometry'])
    lookback_ids = [f['id'] for f in feature_collection_lookback['features']]
    lookback_ids.sort()
    return lookback_ids

def calc_ndvi_minmax(s2_ids, tiles, shape):
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

            ndvi_j = bronco.spectral_index(vir,3,2)
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

def draw_tiled_area(shape, tiles, projection, lonlat_crs, highlights={0:'black'}):
    print 'number of tiles to cover region', len(tiles['features'])
    print tiles['features'][0].keys()

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection=projection) # Specify projection of the map here

    # Get the geometry from each feature
    shapes = [shapely.geometry.shape(tile_j['geometry']) for
              tile_j in tiles['features']]

    ax.add_geometries(shapes, lonlat_crs, color='orange', alpha=0.3)

    ax.add_geometries([shapely.geometry.shape(shape['geometry'])],
                       lonlat_crs, color='blue', alpha=0.7)
    
    for key, value in highlights.iteritems():
        tile = tiles['features'][key]
        ax.add_geometries([shapely.geometry.shape(tile['geometry'])],
                       lonlat_crs, color=value, alpha=0.5)
        print 'tile'+str(key).zfill(3), tile['geometry']

    # Get a bounding box of the combined scenes
    union = shapely.geometry.MultiPolygon(polygons=shapes)
    ax.set_extent((union.bounds[0], union.bounds[2], union.bounds[1], union.bounds[3]), crs=lonlat_crs)
    ax.gridlines(crs=lonlat_crs)
    plt.show()
    
# for Landsat or S2A (aka SENTINEL-2) scenes
def make_label_raster(data_path, place, tile_id, tile, vir_ids, shape,
                  bands=['alpha'], vector_format='geojson'):
    #
    imgfile = data_path+place+'_tile'+str(tile_id).zfill(3)+'_labels'
    print 'imgfile', imgfile
    
    ret = dl.raster.raster(
        vir_ids,
        bands=bands,
        dltile=tile,
        #cutline=shape['geometry'],
        output_format='GTiff',
        data_type='Byte',
        save=True,
        outfile_basename=imgfile
    )
    complete_layer = place.title()+'_Complete'
    if vector_format=='geojson':
        zcomplete = 'OGRGeoJSON'
        os.environ['ZCOMPLETE'] = zcomplete
        zcompleteshp = data_path+complete_layer+'.geojson'
        os.environ['ZCOMPLETESHP'] = zcompleteshp
    if vector_format=='shp':
        zcomplete = complete_layer
        os.environ['ZCOMPLETE'] = zcomplete
        zcompleteshp = data_path+complete_layer+'.shp'
        os.environ['ZCOMPLETESHP'] = zcompleteshp
    zlabels = imgfile+'.tif'
    os.environ['ZLABELS'] = zlabels
    #!gdal_rasterize -a "Land_use" -l $ZCOMPLETE $ZCOMPLETESHP $ZLABELS
    command = 'gdal_rasterize -a Land_use -l {0} {1} {2}'.format(zcomplete,zcompleteshp,zlabels)
    print '>>>',command
    try:
        print subprocess.check_output(command.split(), shell=False)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    
def stats_byte_raster(label_file, category_label, show=False):
    print label_file
    y, ygeo, yprj, ycols, yrows = bronco.load_geotiff(label_file,dtype='uint8')
    yd = {}
    for c in range(256):
        if np.sum((y == c))>0:
            yd[c] = np.sum((y == c))
            print c, yd[c], category_label[c]
    if(show):
        y[y>6]=0  # TBD add a utility function to colorize the result map
        plt.figure(figsize=[8,8])
        plt.imshow(y)
    return yd


def make_osm_raster(data_path, place, tile_id, tile, vir_ids, shape, new_raster=False, burn_value=6,
                  bands=['alpha'], tile_suffix='osm', layer_suffix='_OSM_Roads', vector_format='geojson'):
    #
    imgfile = data_path+place+'_tile'+str(tile_id).zfill(3)+'_'+tile_suffix
    #print 'imgfile', imgfile
    
    if(new_raster):
        ret = dl.raster.raster(
            vir_ids,
            bands=bands,
            dltile=tile,
            #cutline=shape['geometry'], #removed! does not make sense here. now raster all 255 and 1
            output_format='GTiff',
            data_type='Byte',
            save=True,
            outfile_basename=imgfile
        )
    
    osm_layer = place.title()+layer_suffix
    if vector_format=='geojson':
        zosm = 'OGRGeoJSON'
        os.environ['ZOSM'] = zosm
        zosmshp = data_path+osm_layer+'.geojson'
        os.environ['ZOSMSHP'] = zosmshp
    if vector_format=='shp':
        zosm = osm_layer
        os.environ['ZOSM'] = zosm
        zosmshp = data_path+complete_layer+'.shp'
        os.environ['ZOSMSHP'] = zosmshp
    zosmimg = imgfile+'.tif'
    os.environ['ZOSMIMG'] = imgfile+'.tif'
    #print 'gdal_rasterize -burn 6 -l {0} {1} {2}'.format(zosm,zosmshp,zosmimg)
    command = 'gdal_rasterize -burn {3} -l {0} {1} {2}'.format(zosm,zosmshp,zosmimg,burn_value)
    print '>>>',command
    print subprocess.check_output(command.split(), shell=False)

def scale_learning_data(X_train, X_valid):
    scaler = StandardScaler()
    scaler.fit(X_train)
    # apply scaler
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    print X_train_scaled.shape, X_valid_scaled.shape
    # print X_train_scaled[19,:]
    return X_train_scaled, X_valid_scaled, scaler
    
    
def train_model_svm(X_train_scaled, X_valid_scaled, Y_train, Y_valid, categories, alpha=0.003,penalty='l2'):
    model = SGDClassifier(alpha=alpha,penalty=penalty) 
    model.fit(X_train_scaled, Y_train)
    ## evaluate model
    print "evaluate training"
    Yhat_train = model.predict(X_train_scaled)
    confusion = bronco.confusion(Yhat_train,Y_train,categories)
    print "evaluate validation"
    Yhat_valid = model.predict(X_valid_scaled)
    confusion = bronco.confusion(Yhat_valid,Y_valid,categories)
    return Yhat_train, Yhat_valid, model

def spectral_index_tile(img,a,b,tol=1e-6):
    # returns (a - b)/(a + b)
    a_minus_b = np.add(img[a,:,:],np.multiply(img[b,:,:],-1.0))
    a_plus_b = np.add(np.add(img[a,:,:],img[b,:,:]),tol)
    y = np.divide(a_minus_b,a_plus_b)
    y = np.clip(y,-1.0,1.0)
    a_minus_b = None
    a_plus_b = None
    return y

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



# FUNCTION BLOCK: dataset construction
# actual training data file construction and saving (tile-wise)

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


def write_multiband_geotiff(outfile, img, geotrans, prj, data_type=gdal.GDT_Byte):
    driver = gdal.GetDriverByName('GTiff')
    bands = img.shape[0]
    rows = img.shape[1]
    cols = img.shape[2]
    outds = [ ]
    if True:# (data_type == gdal.GDT_Byte):
        opts = ["INTERLEAVE=BAND", "COMPRESS=LZW", "PREDICTOR=1", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"]
        outds  = driver.Create(outfile, cols, rows, bands, data_type, options=opts)
    else:
        outds  = driver.Create(outfile, cols, rows, bands, data_type)
    outds.SetGeoTransform(geotrans)
    outds.SetProjection(prj)
    for b in range(bands):
        outds.GetRasterBand(b+1).WriteArray(img[b])
    del outds
        
def confusion(Yhat,Y,categories):
    n_categories = len(categories)
    confusion = np.zeros((n_categories,n_categories),dtype='uint32')
    for j in range(n_categories):
        for i in range(n_categories):
            confusion[j,i] = np.sum(np.logical_and((Yhat==categories[i]),(Y==categories[j])))
            # print j,i, confusion[j,i], categories[j], categories[i] 
        print categories[j], np.sum(confusion[j,:])
    conf_str = ''
    end_str = ''
    for j in range(n_categories):
        for i in range(n_categories):
            conf_str += str(confusion[j,i]) + '\t'
        #conf_str += str(confusion[j]) + ' '
        conf_str += str(round(float(confusion[j,j]) / float(np.sum(confusion[j,:])),3)) + '\n'
        end_str +=  str(round(float(confusion[j,j]) / float(np.sum(confusion[:,j])),3)) + '\t'
    conf_str += end_str
    #print np.array_str(confusion)
    print conf_str
    
    print confusion.sum(), confusion.trace(), confusion.trace()/float(confusion.sum())
    return confusion
        
        
        
        
        
        
        
        
        
        
        
        
        
        