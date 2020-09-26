import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS
import glob
import os
import time
import math
import itertools
import numpy as np
# raw LiDAR dimensions: 46728, 86749

def get_diff(a, b):
	return a[0]-b[0], a[1]-b[1]

start_time = time.clock()
in_fp = "/mnt/c/users/christel/Documents/THESIS/SecondSem/lidar_viz/tiled_256"
out_fp = "/mnt/c/users/christel/Documents/THESIS/SecondSem/lidar_viz/merged_tiles"
filepaths = glob.glob(os.path.join(in_fp, '*.tif'))
filenames = [x.split('/')[-1].split('.')[0] for x in filepaths]

cols = 1536				# change to 46728
rows = 1536				# change to 86749
tile_size = 768
new_cols = math.ceil(cols/tile_size)
new_rows = math.ceil(rows/tile_size)

for i in range(0, cols, tile_size):
	nums = list(range(i, i+tile_size, 256))
	
	for j in range(0, rows, tile_size):
		nums2 = list(range(j, j+tile_size, 256))
		labels = list(itertools.product(nums, nums2))
		fp_list = []

		for idx, k in enumerate(labels):
			name = 'tile_' + str(k[0]) + '_' + str(k[1])
			if name in filenames:
				fp_list.append(in_fp + '/' + name + '.tif')
				filenames.remove(name)
			else:
				prev = labels[idx-1]
				diff = get_diff(prev, k)

				prev_name = 'tile_' + str(prev[0]) + '_' + str(prev[1])
				prev_fp = in_fp + '/' + prev_name + '.tif'
				prev_tile = rasterio.open(prev_fp)
				
				arr = np.zeros((256, 256), dtype="float32")
				x,y = prev_tile.bounds.left + diff[0], prev_tile.bounds.top + diff[1]
				new_transform = rasterio.transform.from_origin(x, y, 1, 1)
				new_tile = name + '.tif'
				new_meta = prev_tile.meta.copy()
				new_meta.update({"driver": "GTiff",
					"height": arr.shape[0],
					"width": arr.shape[1],
					"transform": new_transform,
					"crs": CRS.from_epsg(32651),
					})

				with rasterio.open(os.path.join(in_fp, new_tile), "w", **new_meta) as dest:
					dest.write(arr, 1)
				# print("Writing: ", new_tile)

		to_mosaic = []
		for fp in fp_list:
			src = rasterio.open(fp)
			to_mosaic.append(src)

		mosaic, out_trans = merge(to_mosaic)
		out_meta = src.meta.copy()
		out_meta.update({"driver": "GTiff",
			"height": mosaic.shape[1],
			"width": mosaic.shape[2],
			"transform": out_trans,
			"crs": CRS.from_epsg(32651),
			})

		new_file = 'tile_' + str(i) + '_' + str(j) + '.tif'
		with rasterio.open(os.path.join(out_fp, new_file), "w", **out_meta) as dest:
			dest.write(mosaic)
		print("Saving: ", new_file)
		print(mosaic.shape)

print('Processor Time: ', time.clock() - start_time, 'seconds')