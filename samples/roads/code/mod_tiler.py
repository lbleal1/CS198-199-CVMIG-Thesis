import os
import gdal
from datetime import datetime
import argparse
import shutil

import cv2
import numpy as np

import tif2png
# considers removing the 50% black

		
def tiler( in_tifname, ftype, tile_size):
	in_path = "/home/lois/Desktop/raw_images/"  
	out_png_path = "/media/lois/LOIS LEAL EXT/lidar/tile_lidar_png/"
	output_folder = "/media/lois/LOIS LEAL EXT/lidar/tiled_lidar/"
	output_filename = 'tile_'
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
		com_string = "gdal_translate -of GTIFF -srcwin " + str(tile_size_x)+ " " + str(tile_size_y) + " " + str(tile_size) + " " + str(tile_size) + " \"" + str(in_path) + str(in_tifname) + "\" " + "\""+ str(output_folder) + str(output_filename) + "\"" + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"
		os.system(com_string)  
		print("tiling = " + str(counter) + " : " + "tile_size_x= " + str(tile_size_x) + "/" + str(xsize) + " : " + "tile_size_y= " + str(tile_size_y) + "/" + str(ysize))

		img_path = output_folder + output_filename + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"
		img_name = output_filename + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"	
		#print(img_name)	
		save_counter += 1
		print("save_counter=", save_counter)
		counter +=1
		tile_size_y += tile_size
	print("Done Tiling")

tiler("ilcsnrt_hillshade.tif", "gmaps", 256)
tif2png.tif2png(output_folder, out_png_path )
