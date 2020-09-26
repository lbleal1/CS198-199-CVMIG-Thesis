from helpers import *
from datetime import datetime
import pandas as pd
import numpy as np
import math
from datetime import datetime
import rasterio
def main():
	prelims = {}
	dict_runs = {}
	seg_runs = {}
	raw_lidar_unroll_runs = {}
	erase_runs = {}
	tiffer_saver_runs = {}
	seg_save_runs = {}
	dictios = []
	dictios_names = ["prelims", "dict_runs", "seg_runs", "raw_lidar_unroll_runs", "erase_runs", "tiffer_saver_runs" ]

	# data settings 
	src_type = "lidar" # the other is ifsar
	if src_type == "lidar":
		input_adf = "dtm_z1001.adf"
		output_tif = "dtm_z1001.tif"
	else:
		output_tif = "dtm_z1001.tif"

	# 1. tiling and saving paths+fnames to csv
	# - gmaps
	gmaps_time = datetime.now()

	# tile gmaps
	tile_gmaps_time = datetime.now()
	tiler(in_tifname = "gmaps2.tif", ftype = "gmaps", tile_size = 256)
	prelims["tile_gmaps_time"] = datetime.now() - tile_gmaps_time
	# get only the three channels
	threeChannels_gmaps_time = datetime.now()
	toThreeChannels("tiled_gmaps", "tiled_gmaps_3")
	prelims["threeChannels_gmaps_time"] = datetime.now() - threeChannels_gmaps_time 
	# files to csv
	files_gmaps_time = datetime.now()
	outpath_csv_gmaps3, out_csv_gmaps3 = files2csv("tiled_gmaps_3", "sat_road_local.csv")
	prelims["files_gmaps_time"] = datetime.now() - files_gmaps_time
	#
	prelims["gmaps_time"] = datetime.now() - gmaps_time

	# - lidar data or ifsar data preprocess
	if src_type == 'lidar':
		adf2tif_lidar_time = datetime.now()
		adf2tif(input_adf, output_tif)
		prelims["adf2tif_lidar_time"] = datetime.now() - adf2tif_lidar_time

	# tile
	tile_lidar_time = datetime.now()
	tiler(in_tifname = "dtm_z1001.tif", ftype = "lidar", tile_size = 256)
	prelims["tile_lidar_time"] = datetime.now() - tile_lidar_time
	# file to csv
	files_lidar_time = datetime.now()
	outpath_csv_lidar, outcsv = files2csv(in_folder = "tiled_raw_lidar", out_csv = "lidar_road_local.csv")
	prelims["files_lidar_time"] = datetime.now() - files_lidar_time
	print("putting tiled_raw_lidar to csv done:", prelims["files_lidar_time"])


	# 2. Segmenting
	# - set batch runs
	batch_runs = 11
	main_start=datetime.now()
	print(outpath_csv_gmaps3)

	ds = pd.read_csv(outpath_csv_gmaps3)
	data_num = len(ds['file_path'])

	for run in range(1, batch_runs+1):
	    run_time=datetime.now()

	    # - set start stop
	    start = (math.ceil(data_num/batch_runs))*(run-1)
	    stop = ((math.ceil(data_num/batch_runs))*run)-1
	    if stop > data_num:
	    	stop = data_num

	    # - get segmented result
	    seg_time = datetime.now()
	    seg_result, sat_result = segmenter(run, data_num, "tiled_gmaps_3", "sat_road_local.csv", start, stop)
	    seg_runs[run] = datetime.now()-seg_time
	    
	    # 3. Erasing
	    # - get raw lidar data unrolled
	    raw_lidar_df = pd.read_csv(outpath_csv_lidar)
	    # -- batch the raw_lidar tiles
	    new_raw_lidar_df = raw_lidar_df[start:stop]
	    new_raw_lidar_df.reset_index(drop=True, inplace=True)
	    raw_lidar_unroll_time = datetime.now()
			#
	    counter = 0
	    raw_lidar_unrolled = []
	    for i in new_raw_lidar_df['file_path']:
	    	raw_lidar_unrolled.append(np.squeeze(rasterio.open(i).read()))
	    	print("raw lidar unrolling = " + str(counter))
	    	counter+=1
	    raw_lidar_unroll_runs[run] = datetime.now() - raw_lidar_unroll_time 
	    
	    # - erasing
	    erase_time = datetime.now()
	    erased_lidar = eraser(seg_result,raw_lidar_unrolled, run)
	    erase_runs[run] = datetime.now() - erase_time

	    # 4. Tiffing and Saving
	    tiffer_saver_time = datetime.now()
	    tiffer_saver(erased_lidar,new_raw_lidar_df,run)
	    tiffer_saver_runs[run] = datetime.now() - tiffer_saver_time
	    
			# convert segresult (list) to numpy array
	    segresult = np.asarray(segresult)
	    if str(type(seg_result)) == "<class 'numpy.ndarray'>":  # if unet this is list  
	    	seg_save_time = datetime.now()
	    	seg_save(seg_result, new_raw_lidar_df, run)
	    	seg_save_runs[run] = datetime.now() - seg_save_time

	    	dictios.append(seg_save_runs)
	    	dictios_names.append("seg_save_runs")

	# 5. Stitching
	if src_type == "lidar":
		stitcher_time = datetime.now()
		print("stitching...")
		stitcher(in_folder = "tiled_erased_lidar", out_folder = "stitched_lidar_1280", tile_size = 1280, orig_tif = "../../assets/raw_data/dtm_z1001.tif")
	# if ifsar, do nothing, we want 256x256
		print("stitcher_time =", datetime.now() - stitcher_time )	
	
	dict_runs[run] = datetime.now()-run_time
	
	print("dictionary...")
	dictios = [prelims, dict_runs, seg_runs, raw_lidar_unroll_runs, erase_runs, tiffer_saver_runs ]
	read_dict(dictios_names, dictios)

if __name__=="__main__": 
	main()
	print("Im so done")