import numpy as np
import subprocess
import pandas as pd
import os.path

import util_rasters


def generate_chips(data_root, place, tiles,
				   label_suffix, label_lot,
				   source, image_suffix, bands,
				   resampling='bilinear', processing_level=None,
				   chip_radius=32,
				   remove_duplicates=True,
				   category_label={0:'Open Space',1:'Non-Residential',\
					   2:'Residential Atomistic',3:'Residential Informal Subdivision',\
					   4:'Residential Formal Subdivision',5:'Residential Housing Project',\
					   6:'Roads',7:'Study Area',8:'Labeled Study Area',254:'No Data',255:'No Label'},
				   show_stats=False,
				   tile_start=None,
				   tile_stop=None
				  ):

	resolution = int(tiles['features'][0]['properties']['resolution'])
	size = int(tiles['features'][0]['properties']['tilesize'])
	pad = int(tiles['features'][0]['properties']['pad'])

	if resolution==10:
		zfill=3
	elif resolution==5:
		zfill=4
	elif resolution==2:
		zfill=5	
	else:
		raise Exception('bad resolution: '+str(resolution))
		

	rows_list = []
	# select city
	# loop through ground truth tiles
	if tile_start is None:
		tile_start = 0;
	if tile_stop is None:
		tile_stop = len(tiles['features'])
	for tile_id in range(tile_start, tile_stop):
		# for each tile:
			# sample file name: /data/phase_iv/sitapur/gt/sitapur_aue0_5m_p32_tile0586_lulc.tif
		path_base = data_root+place+'/gt/'+place+'_'+label_suffix+label_lot+'_'+str(resolution)+'m'+'_'+\
			'p'+str(pad)+'_'+'tile'+str(tile_id).zfill(zfill)
		#print path_base
		path_lulc = path_base+'_'+'lulc.tif'
		path_locale = path_base+'_'+'locale.tif'
		lulc,_,_,_,_ = util_rasters.load_geotiff(path_lulc,dtype='uint8')
		locale,_,_,_,_ = util_rasters.load_geotiff(path_locale,dtype='uint8')
		# 'erase' irrelevant pixels
		lulc[0:pad,:] = 255; lulc[-pad:,:] = 255; lulc[:,0:pad] = 255; lulc[:,-pad:] = 255
		locale[0:pad,:] = 255; locale[-pad:,:] = 255; locale[:,0:pad] = 255; locale[:,-pad:] = 255
		# get locations of target pixels
		locs = np.where(lulc!=255)
		n_px = len(locs[0])
		if n_px == 0: 
			continue
		if show_stats:
			util_rasters.stats_byte_raster(lulc, category_label, lulc=True, show=True)
		print place + ' '+ image_suffix + ': non-pad, labeled pixels in tile'+str(tile_id).zfill(zfill)+':', len(locs[0])
		#image path example: /data/phase_iv/sitapur/imagery/none/sitapur_s2_E_5m_p32_tile0006.tif
		path_image = data_root+place+'/imagery/'+str(processing_level).lower()+'/'+place+'_'+source+'_'+\
			image_suffix+'_'+str(resolution)+'m'+'_'+'p'+str(pad)+'_'+'tile'+str(tile_id).zfill(zfill)+'.tif'
		# loop through list of locations of target pixels
		for i in range(len(locs[0])):
			row = locs[0][i]
			col = locs[1][i]
			#print row,col,lulc[row,col]
			# for each pixel:
			# grab a window of imagery centered around target pixel
			xoff = col - chip_radius; yoff = row - chip_radius;
			xsize = chip_radius*2+1; ysize = chip_radius*2+1
			# save this as a separate "image chip" geotiff, with careful naming
			path_chip = data_root+place+'/chips/'+str(processing_level).lower()+'/'+\
				place+'_'+label_suffix+label_lot+'_'+source+'_'+image_suffix+'_'+str(resolution)+'m'+'_'+\
				't'+str(tile_id).zfill(zfill)+'_'+'x'+str(col-pad).zfill(3)+'y'+str(row-pad).zfill(3)+'_'+\
				'c'+str(lulc[row,col])+'.tif'
			#print path_image
			#print path_chip
			#print xoff, yoff, xsize, ysize
			#gdal template: gdal_translate -srcwin xstart ystart xstop ystop input.raster output.raster
			#!gdal_translate -q -srcwin {xoff} {yoff} {xsize} {ysize} {path_image} {path_chip}
			command = 'gdal_translate -q -srcwin {0} {1} {2} {3} {4} {5}'.format(xoff,yoff,xsize,ysize,path_image,path_chip)
			#print '>>>',command
			try:
				subprocess.check_output(command.split(), shell=False)
			except subprocess.CalledProcessError as e:
				raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
			
			# as each chip is saved, add an entry to a catalog dataframe
			#['city','gt_type','gt_lot','locale','source','image','bands',
			#   'resolution','resampling','processing','tile_id','column','row','lulc']
			row_dict = {}
			row_dict['path']=path_chip
			row_dict['city']=place
			row_dict['gt_type']=label_suffix
			row_dict['gt_lot']=label_lot
			row_dict['locale']=locale[row,col]
			row_dict['source']=source
			row_dict['image']=image_suffix
			row_dict['bands']=bands
			row_dict['resolution']=resolution
			row_dict['resampling']=resampling
			row_dict['processing']=str(processing_level).lower()
			row_dict['tile_id']=tile_id
			row_dict['x']=col-pad
			row_dict['y']=row-pad
			row_dict['lulc']=lulc[row,col]
			rows_list.append(row_dict)
			
	columns = ['path','city','gt_type','gt_lot','locale','source','image','bands',
			   'resolution','resampling','processing','tile_id','x','y','lulc']
	df_new = pd.DataFrame(rows_list, columns=columns)
	#DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
	#df_new.set_index('path',drop=True,append=False,inplace=True,verify_integrity=True)
	df_new.head()
	# save dataframe
	path_catalog = data_root+'chip_catalog.csv'
	if not os.path.isfile(path_catalog):
		#write new records directly
		df_new.to_csv(path_catalog,index=False,header=True)
	else:
		#read csv
		#load to dataframe
		df_old = pd.read_csv(path_catalog)
		# append new
		#DataFrame.append(other, ignore_index=False, verify_integrity=False, sort=None)
		df_combo = df_old.append(df_new,ignore_index=False,verify_integrity=False)
		#remove duplicates
		if remove_duplicates:
			#DataFrame.drop_duplicates(subset=None, keep='first', inplace=False)
			df_combo.drop_duplicates(subset='path',keep='first',inplace=True)
		df_combo.to_csv(path_catalog,index=False,header=True)

def load_catalog(data_root='/data/phase_iv/',catalog_name='chip_catalog'):
	path_catalog = data_root+catalog_name+'.csv'
	return pd.read_csv(path_catalog)