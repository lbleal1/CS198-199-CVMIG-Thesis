import os
import gdal
import time
import glob

start_time = time.clock()

src_dir = "tiled"
in_path = "/media/riza/'Seagate Bac'/NEW/lidar_viz/tiled/"
dest_dir = "/media/riza/'Seagate Bac'/NEW/lidar_viz/Hillshaded/"
file_list = glob.glob(os.path.join(src_dir, '*.tif'))

azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

for file in file_list:
	filename = file.split('/')[-1].split('.')[0]

	# one-directional hillshading
	for angle in azimuths:
		com_string = "gdaldem hillshade -compute_edges -az " + str(angle) + " " + str(in_path) + str(filename) + ".tif " + str(dest_dir) + str(angle) + "/" + str(filename) + ".tif"
		os.system(com_string)

	# multi-directional hillshading
	com_string = "gdaldem hillshade -multidirectional -compute_edges " + str(in_path) + str(filename) + ".tif " + str(dest_dir) + "multi/" + str(filename) + ".tif"
	os.system(com_string)

	# slope map
	com_string = "gdaldem slope -compute_edges " + str(in_path) + str(filename) + ".tif " + str(dest_dir) + "slope/" + str(filename) + ".tif"
	os.system(com_string)

	# aspect map
	com_string = "gdaldem aspect -compute_edges " + str(in_path) + str(filename) + ".tif " + str(dest_dir) + "aspect/" + str(filename) + ".tif"
	os.system(com_string)

print('Processor Time: ', time.clock() - start_time, 'seconds')