import descarteslabs as dl


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

	for suffix, ids in image_dict.iteritems():
	    print suffix, ids

	    for tile_id in range(len(tiles['features'])):
	        tile = tiles['features'][tile_id]
	        basename = data_root+place+'/imagery/'+str(processing_level).lower()+'/'+\
	        	place+'_'+source+'_'+suffix+'_'+str(resolution)+'m'+'_'+'p'+str(pad)+'_'+'tile'+str(tile_id).zfill(zfill)
	        print 'downloading tile'+str(tile_id).zfill(zfill)+':', basename+'.tif'
	        vir = dl.raster.raster(
                ids,
                bands=bands,
                resampler=resampler,
                data_type='UInt16',
                dltile=tile,
                cutline=shape['geometry'],
                processing_level=processing_level,
                save=True,
                outfile_basename=basename)