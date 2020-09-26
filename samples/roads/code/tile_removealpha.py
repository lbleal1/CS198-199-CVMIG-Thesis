import os
import gdal
from datetime import datetime
import argparse
import shutil

import cv2
import numpy as np

# doesn't consider removing the 50% black
# since the tiles are already being chosen using the 
# image names in the past gmaps tiles (which are already filtered)

def check_img(img_path, img_name, out_png_path):
	

def get_fnames(src_folder):
			return [fname for fname in os.listdir(src_folder)]
		
def tiler( in_tifname, ftype, tile_size):
	in_path = "../assets/raw_data/"
	
	# gmaps names basis  
	gmaps_path = "../assets/output_data/tiled_gmaps_3"

	out_png_path = "../assets/output_data/hard_lidar/tile_hl_png/"
	output_folder = "../assets/output_data/hard_lidar/tiled_hl/"
	output_filename = 'tile_'

	gmaps = get_fnames(gmaps_path)

	band_number = 1	
	ds = gdal.Open(in_path + in_tifname)
	band = ds.GetRasterBand(band_number)
	xsize = band.XSize
	ysize = band.YSize

	#xsize = 256
	#ysize = 256

	tile_size_y = 0
	tile_size_x = 0

	counter = 0
	save_counter = 0
	while tile_size_x < xsize:
		if tile_size_y > ysize:
			tile_size_y = 0
			tile_size_x += tile_size
		
		img_name = output_filename + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"
		img_path = output_folder + output_filename + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"
		print("tiling = " + str(counter) + " : " + "tile_size_x= " + str(tile_size_x) + "/" + str(xsize) + " : " + "tile_size_y= " + str(tile_size_y) + "/" + str(ysize))
		
		if img_name in gmaps:
		# tile then get tif tile
			com_string = "gdal_translate -of GTIFF -srcwin " + str(tile_size_x)+ " " + str(tile_size_y) + " " + str(tile_size) + " " + str(tile_size) + " \"" + str(in_path) + str(in_tifname) + "\" " + "\""+ str(output_folder) + str(output_filename) + "\"" + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"
			os.system(com_string)  
			
		# convert to three channels but still tif
		out_three_path = "../assets/output_data/hard_lidar/tiled_hl_3/" + img_name
		os.system("gdal_translate -b 1 -b 2 -b 3 " + img_path + " " + out_three_path) 				

			save_counter +=1
			print("save_counter=", save_counter)
		else:
			print("not saved")

		counter +=1
		tile_size_y += tile_size
	print("Done Tiling")

tiler("ilcs_ifsar_multi.tif", "gmaps", 256)
