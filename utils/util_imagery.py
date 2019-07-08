import descarteslabs as dl
import utils.util_rasters as util_rasters
import numpy as np
import os
import gdal

def download_imagery(data_root, place, source, bands, shape, tiles, image_dict, 
		resampler='bilinear', processing_level=None):
	resolution = int(tiles['features'][0]['properties']['resolution'])
	pad   = int(tiles['features'][0]['properties']['pad'])
	if resolution==10: 
	    zfill=3
	elif resolution==5:
	    zfill=4
	elif resolution==2:
	    zfill=5    
	else:
	    raise Exception('bad resolution: '+str(resolution))

	for suffix, ids in image_dict.items():
	    print(suffix, ids)

	    for tile_id in range(len(tiles['features'])):
	        tile = tiles['features'][tile_id]
	        basename = data_root+place+'/imagery/'+str(processing_level).lower()+'/'+\
	        	place+'_'+source+'_'+suffix+'_'+str(resolution)+'m'+'_'+'p'+str(pad)+'_'+'tile'+str(tile_id).zfill(zfill)
	        print('downloading tile'+str(tile_id).zfill(zfill)+':', basename+'.tif')
	        vir = dl.raster.raster(
                ids,
                bands=bands,
                resampler=resampler,
                data_type='UInt16',
                dltile=tile,
                cutline=shape['geometry'], # how can she cut
                processing_level=processing_level,
                save=True,
                outfile_basename=basename)

def s2_preprocess(im, bands_first=False):
    # probably don't need to drop alpha..
    # drop alpha
    if bands_first:
        im = im[:-1,:,:]
    else:
        im = im[:,:,:-1]
    # manual "rescaling" as in all previous phases
    im = im.astype('float32')
    im = im/10000.
    im = np.clip(im,0.0,1.0)
    #print('im prepped')
    return im

def spectral_index(img,a,b,tol=1e-6,bands_first=False):
    # returns (a - b)/(a + b)
    if bands_first:
        a_minus_b = np.add(img[a,:,:],np.multiply(img[b,:,:],-1.0))
        a_plus_b = np.add(np.add(img[a,:,:],img[b,:,:]),tol)
    else:
        a_minus_b = np.add(img[:,:,a],np.multiply(img[:,:,b],-1.0))
        a_plus_b = np.add(np.add(img[:,:,a],img[:,:,b]),tol)
    y = np.divide(a_minus_b,a_plus_b)
    y = np.clip(y,-1.0,1.0)
    a_minus_b = None
    a_plus_b = None
    return y

#needs bands_first update if to be used in future
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
        print("not enough dark pixels for haze correction")
        return img, refl_offsets
    #
    if n_dark_pix>1e5:
        ds = float(int(n_dark_pix/1e5))
    else:
        ds = 1.
    n_dark_pix /= ds
    print('n_dark_pixels: %i' % n_dark_pix)
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
        print(b, offsets[b], refl_offsets[b], img[b].min(), corrected_img[b].min(), corrected_img[b].max())
    #
    return corrected_img, refl_offsets

def s2_cloud_mask(X,get_rgb=False, 
                  key={0:'nodata',1:'land',2:'snow',3:'water',4:'clouds',5:'vegetation',6:'shadows'},
                  bands_first=False):
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
    if bands_first:
        L1 = X[0,:,:]
        L2 = X[1,:,:]
        L3 = X[2,:,:]
        L5 = X[4,:,:]
        alpha = X[-1,:,:]
    else:
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

    ndsi = spectral_index(X,1,4,bands_first=bands_first)
    Y[(ndsi > 0.4)]  = 2  
    #
    ndvi = spectral_index(X,3,2,bands_first=bands_first)
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
    grey = spectral_index(X,1,0,bands_first=bands_first)
    index = np.logical_and(index, (abs(grey) < 0.2))    
    grey = spectral_index(X,2,1,bands_first=bands_first)
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

def ls_cloud_mask(X,T,get_rgb=False, 
               key={0:'nodata',1:'land',2:'snow',3:'water',4:'clouds',5:'vegetation',6:'shadows'},
               bands_first=False):
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
    if bands_first:
        L1 = X[0,:,:]
        L2 = X[1,:,:]
        L3 = X[2,:,:]
        L5 = X[4,:,:]
        alpha = X[-1,:,:]
    else:
        L1 = X[:,:,0]
        L2 = X[:,:,1]
        L3 = X[:,:,2]
        L5 = X[:,:,4]
        alpha = X[:,:,-1]
    L6 = T   # celsius
    #
    # land - default value
    #
    Y = np.ones(L1.shape,dtype='uint8')
    #
    # snow/ice
    #
    ndsi = spectral_index(X,1,4,bands_first=bands_first)
    #
    # Y[(L6 <= 0)] = 2  # v2
    Y[np.logical_or((L6 <= 0),(ndsi > 0.4))]  = 2  
    #
    ndvi = spectral_index(X,3,2,bands_first=bands_first)
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
    grey = spectral_index(X,1,0,bands_first=bands_first)
    index = np.logical_and(index, (abs(grey) < 0.2))    
    grey = spectral_index(X,2,1,bands_first=bands_first)
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

# bands first/last irrelevant, since returns stack of 2D arrays
def calc_index_minmax(ids, tiles, shape, band_a, band_b,
                    bands=['blue','green','red','nir','swir1','swir2','alpha']):
    ntiles = len(tiles['features'])
    tilepixels = tiles['features'][0]['properties']['tilesize'] + (tiles['features'][0]['properties']['pad']*2)

    max_tiles = np.full([ntiles, tilepixels, tilepixels], np.nan, dtype='float32')
    min_tiles = np.full([ntiles, tilepixels, tilepixels], np.nan, dtype='float32')

    for j in range(len(ids)):
        print('image', j)
        for tile_id in range(ntiles):
            if(tile_id % 50 == 0):
               print('    tile', tile_id)
            tile = tiles['features'][tile_id]

            vir, vir_meta = dl.raster.ndarray(
                ids[j],
                bands=bands,
                data_type='UInt16',
                dltile=tile,
                #cutline=shape['geometry'] # removing to try to sidestep nan issue
            )
            #print(vir.shape)
            vir = vir.astype('float32')
            vir = vir/10000.
            vir = np.clip(vir,0.0,1.0)

            ndvi_j = spectral_index(vir,band_a,band_b,bands_first=False)
            masking, key = s2_cloud_mask(vir,bands_first=False)

            if (j==0):
                ndvi_max = np.full([ndvi_j.shape[0], ndvi_j.shape[1] ], np.nan, dtype=vir.dtype)
                ndvi_min = np.full([ndvi_j.shape[0], ndvi_j.shape[1] ], np.nan, dtype=vir.dtype)

            alpha_j = vir[:,:,-1]
            cloudfree_j = (masking != 4)
            goodpixels_j = np.logical_and(alpha_j, cloudfree_j)

            np.fmax(ndvi_j, max_tiles[tile_id], out=max_tiles[tile_id], where=goodpixels_j)
            np.fmin(ndvi_j, min_tiles[tile_id], out=min_tiles[tile_id], where=goodpixels_j)

    print('done')
    return min_tiles, max_tiles


def show_vir(file,
            bands_first=True):
    img, geo, prj, cols, rows = util_rasters.load_geotiff(file)
    assert img.ndims==3
    if bands_first:
        img = np.transpose(img, (1,2,0))
    img = img[:,:,0:3]
    img = np.flip(img, axis=-1)

    viz = img.astype('float32')
    viz = viz/3000 # why 3000?
    viz = 255.*viz
    viz = np.clip(viz,0,255)
    viz = viz.astype('uint8')
    for b in range(img.shape[2]):
        print(b, np.min(viz[:,:,b]), np.max(viz[:,:,b]))
    plt.figure(figsize=[16,16])
    plt.imshow(viz)

def calc_water_mask(vir, idx_green=1, idx_nir=3, threshold=0.15, bands_first=False):
    if bands_first:
        assert vir.shape[0]==6
    else:
        assert vir.shape[2]==6

    band_a = idx_green
    band_b = idx_nir
    threshold = threshold # water = ndwi > threshold 
    ndwi = spectral_index(vir, band_a, band_b, bands_first=bands_first)
    water = ndwi > threshold

    return water


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

def rgb_lulc_result(Y,BIP=True,
                    color_scheme={
                        0:"b2df8a", # open space
                        1:"fb9a99", # non-residential
                        2:"ff7f00", # residential - atomistic
                        3:"fdbf6f", # residential - informal
                        4:"1f78b4", # residential - formal
                        5:"a6cee3", # residential - projects
                        6:"e31a1c", # roads
                        9:"003366", # water
                        254:"000000", # outside study area
                        255:"f3f3f3", # no data
                    },
                    ):
    rgb = np.zeros((3,Y.shape[0],Y.shape[1]),dtype='uint8')
    for k, v in color_scheme.items():
        rgb[0][(Y==k)] = int(v[0:2], 16)
        rgb[1][(Y==k)] = int(v[2:4], 16)
        rgb[2][(Y==k)] = int(v[4:6], 16)
    
    if (BIP==True): # flips axes; don't know what BIP stands for
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
