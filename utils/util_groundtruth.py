
import descarteslabs as dl
import subprocess
import os


# creating new rasters from tiles and vector info
# for Landsat or S2A (aka SENTINEL-2) scenes
def make_label_raster(data_path, place, tile_id, tile, vir_ids, shape,
                  label_lot='0', label_suffix='aue', vector_format='geojson',
                  burn_attribute='Land_use', touch_category=None):
    #
    bands=['alpha']
    resolution = int(tile['properties']['resolution'])
    size = int(tile['properties']['tilesize'])
    pad = int(tile['properties']['pad'])

    if resolution==10:
        zfill=3
    elif resolution==5:
        zfill=4
    elif resolution==2:
        zfill=5    
    else:
        raise Exception('bad resolution: '+str(resolution))

    if burn_attribute=='Land_use':
        tag = 'lulc'
    elif burn_attribute=='Locale_No':
        tag = 'locale'
    else:
        raise Exception('bad burn_attribute: '+burn_attribute)
    
    #imgfile = data_path+place+'_tile'+str(tile_id).zfill(zfill)+'_'+label_suffix+('' if resolution==10 else ('_'+str(resolution))+'m')
    # imgfile = data_path+place+'_'+'l'+label_lot+'_'+label_suffix+'_'+str(resolution)+'m'+'_'+'p'+str(pad)+'_'+'tile'+str(tile_id).zfill(zfill)
    imgfile = data_path+'gt/'+place+'_'+label_suffix+label_lot+'_'+str(resolution)+'m'+'_'+'p'+str(pad)+'_'+'tile'+str(tile_id).zfill(zfill)+'_'+tag
    print('imgfile', imgfile)
    
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
    if label_lot != '0': # envisioning alternate ground-truth file would be named eg "Place_Complete_2"
        complete_layer += '_'+label_lot
    if vector_format=='geojson':
        zcomplete = 'OGRGeoJSON'
        os.environ['ZCOMPLETE'] = zcomplete
        zcompleteshp = data_path+'gt/'+complete_layer+'.geojson'
        os.environ['ZCOMPLETESHP'] = zcompleteshp
    if vector_format=='shp':
        zcomplete = complete_layer
        os.environ['ZCOMPLETE'] = zcomplete
        zcompleteshp = data_path+'gt/'+complete_layer+'.shp'
        os.environ['ZCOMPLETESHP'] = zcompleteshp
    zlabels = imgfile+'.tif'
    os.environ['ZLABELS'] = zlabels

    #!gdal_rasterize -a "Land_use" -l $ZCOMPLETE $ZCOMPLETESHP $ZLABELS
    #command = 'gdal_rasterize -a Land_use -l {0} {1} {2}'.format(zcomplete,zcompleteshp,zlabels)
    command = 'gdal_rasterize -a {2} {0} {1}'.format(zcompleteshp,zlabels,burn_attribute)
    print('>>>',command)
    try:
        s=0
        print(subprocess.check_output(command.split(), shell=False),'\n')
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    
    if touch_category is None:
        return
    cat = int(touch_category)
    if cat > 255 or cat < 0:
        raise ValueError('Illegal touch_category passed to make_label_raster: '+touch_category)
        return
    command = 'gdal_rasterize -a {3} -where {3}=\'{2}\' -at {0} {1}'.format(zcompleteshp,zlabels,cat,burn_attribute)
    print('>>>',command)
    try:
        s=0
        print(subprocess.check_output(command.split(), shell=False),'\n')
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))