# organized set of helper functions
# drawn from bronco.py and bronco notebooks
# topic: descartes labs api

import warnings
warnings.filterwarnings('ignore')
#
import os
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
import matplotlib.pyplot as plt
#
import descarteslabs as dl
import subprocess

# GENERAL

def place_slug(*parts):
    return "_".join(parts)

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
    print('shape:', arr.shape)
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
    print('shape:', arr.shape)
    # pprint(meta)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    plt.imshow(arr)
    return

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
        print(bands_info[0].keys())
        # import operator
        # bands_info.sort(key=operator.itemgetter('name'))
        for b in range(len(bands_info)):
            band_info = bands_info[b]
            print(b, band_info['name'], band_info['description'])
            try: 
                print(band_info['name_vendor'], band_info['vendor_order'])
            except:
                print('derived band')
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
                if(not silent): print(b, band_info['name'], 'add')
        except:
            if(not silent): print(b, band_info['name'], 'skip')
            continue
    print() 
    bands = [ u'coastal-aerosol', u'blue', u'green', u'red',  u'nir',  u'swir1', u'swir2'] # S2
    for key in bands:
        try:
            if(not silent): print(vendor_bands_info[key]['name'], vendor_bands_info[key]['name_vendor'])
            if(not silent): print('physical_range', vendor_bands_info[key]['physical_range'])
            if(not silent): print('valid_range', vendor_bands_info[key]['data_range'])
            if(not silent): print('nbits', vendor_bands_info[key]['nbits'])
            if(not silent): print('resolution', vendor_bands_info[key]['resolution'])
        except:
            if(not silent): print('absent for this satellite')
        if(not silent): print()
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
        print(start_time, end_time)

    feature_collection_lookback = dl.metadata.search(sat_id=satellites, start_time=start_time, end_time=end_time, 
                                            cloud_fraction_0=cloud_fraction_0, limit=limit, geom=shape['geometry'])
    lookback_ids = [f['id'] for f in feature_collection_lookback['features']]
    lookback_ids.sort()
    return lookback_ids

    
def make_osm_raster(data_path, place, tile_id, tile, vir_ids, shape, new_raster=False, burn_value=6,
                  bands=['alpha'], tile_suffix='osm', layer_suffix='_OSM_Roads', vector_format='geojson'):
    #
    imgfile = data_path+place+'_tile'+str(tile_id).zfill(3)+'_'+tile_suffix
    #print('imgfile', imgfile)
    
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
    #print('gdal_rasterize -burn 6 -l {0} {1} {2}'.format(zosm,zosmshp,zosmimg))
    command = 'gdal_rasterize -burn {3} -l {0} {1} {2}'.format(zosm,zosmshp,zosmimg,burn_value)
    print('>>>',command)
    print(subprocess.check_output(command.split(), shell=False))


    