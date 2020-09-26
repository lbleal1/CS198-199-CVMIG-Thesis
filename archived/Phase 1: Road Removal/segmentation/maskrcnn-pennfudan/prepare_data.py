import os
import gdal
from datetime import datetime
import argparse
import shutil

def tiler(in_tif, tile_size, counter, output_folder, band_number):
  output_filename = 'tile_'

  ds = gdal.Open(in_tif)
  band = ds.GetRasterBand(band_number)
  xsize = band.XSize
  ysize = band.YSize

  tile_size_y = 0
  tile_size_x = 0


  while tile_size_x < xsize:
    if tile_size_y > ysize:
      tile_size_y = 0
      tile_size_x += tile_size
    if band_number == 1:
      com_string = "gdal_translate -of GTIFF -srcwin " + str(tile_size_x)+ " " + str(tile_size_y) + " " + str(tile_size) + " " + str(tile_size) + " \"" + str(in_tif) + "\" " + "\""+ str(output_folder) +  str(output_filename) + "\"" + str(counter) + "_mask" + ".tif"
    else:
      com_string = "gdal_translate -of GTIFF -srcwin " + str(tile_size_x)+ " " + str(tile_size_y) + " " + str(tile_size) + " " + str(tile_size) + " \"" + str(in_tif) + "\" " + "\""+ str(output_folder) +  str(output_filename) + "\"" + str(counter)  + ".tif"
    os.system(com_string)  
    tile_size_y += tile_size
    print("tiling = " + str(counter))
    counter+=1
  return counter


def main():
  src_folder = "mass_roads"
  folders = ["/train", "/valid", "/test"]

  counter = 0
  for folder in folders:
    fpath_1 = src_folder + folder + '/map/'
    for fname in sorted(os.listdir(fpath_1)):
      print(folder)
      counter_1 = tiler(fpath_1 + fname, 256, counter, "data/masks/", 1)
      counter = counter_1

  counter = 0
  for folder in folders:
    fpath_2 = src_folder + folder + '/sat/'
    for fname in sorted(os.listdir(fpath_2)):
      counter_2 = tiler(fpath_2 + fname, 256, counter, "data/imgs/", 3)
      counter = counter_2
    

if __name__=="__main__": 
	main()
