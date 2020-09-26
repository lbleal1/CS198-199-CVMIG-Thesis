import os
import gdal
from datetime import datetime
import argparse
import shutil

import cv2
import numpy as np
import tif2png

in_path = "../assets/raw_data/"
	
# gmaps names basis  
gmaps_path = "../assets/output_data/tiled_gmaps_3"

out_png_path = "../assets/output_data/lidar/tile_lidar_png/"
output_folder = "../assets/output_data/lidar/tiled_lidar/"
output_filename = 'tile_'

def get_fnames(src_folder):
			return [fname for fname in os.listdir(src_folder)]

def tiler( in_tifname, tile_size):
	gmaps = get_fnames(gmaps_path)

	band_number = 1	
	ds = gdal.Open(in_path + in_tifname)
	band = ds.GetRasterBand(band_number)
	xsize = band.XSize
	ysize = band.YSize

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
			com_string = "gdal_translate -of GTIFF -srcwin " + str(tile_size_x)+ " " + str(tile_size_y) + " " + str(tile_size) + " " + str(tile_size) + " \"" + str(in_path) + str(in_tifname) + "\" " + "\""+ str(output_folder) + str(output_filename) + "\"" + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"
			os.system(com_string)  
			
			#print(img_name)	
			#save_counter += check_img(img_path, img_name, out_png_path)
			save_counter +=1
			print("save_counter=", save_counter)
		else:
			print("not saved")

		counter +=1
		tile_size_y += tile_size
	print("Done Tiling")


tiler("ilcsnrt_lidar.tif", 256)
print("Tiling done.")

print("Converting to png...")
tif2png.tif2png(output_folder, out_png_path )
print("Done.")
