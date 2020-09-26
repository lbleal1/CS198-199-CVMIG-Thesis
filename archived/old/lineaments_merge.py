import os
import glob
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from skimage.draw import line
from affine import Affine
import pandas as pd
import time


def to_tuple(string):
	return tuple(int(a) for a in string.strip('()').split(','))

def draw_lines(img, lines, color):
	for line in lines:
		line = to_tuple(line)
		x1, y1, x2, y2 = line
		cv2.line(img, (x1, y1), (x2, y2), color, 1)
	return


start_time = time.time()
df = pd.read_csv('Lineaments/lineaments_data.csv')
# print(df.head())

src_dir = 'Hillshaded/multi'
dest_dir = 'Lineaments/merged'
file_list = glob.glob(os.path.join(src_dir, '*.tif'))
azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
colors = {
	0: (0, 0, 128),
	45: (0, 128, 0),
	90: (128, 0, 0),
	135: (128, 0, 128),
	180: (128, 128, 0),
	225: (0, 128, 128),
	270: (255, 20, 147),
	315: (0, 165, 255)
}

for file in file_list:
	# print(file)
	with rio.open(file) as dtm:
		dtm_arr = dtm.read(1, masked=True)
		dtm_rgb = cv2.cvtColor(dtm_arr, cv2.COLOR_GRAY2RGB)
		dim = (dtm.width * dtm.height) // 2

		if np.ma.count_masked(dtm_arr) < dim:
			filename = file.split('/')[-1]
			for angle in azimuths:
				lines = df[(df['filename'] == filename) & (df['azimuth'] == angle)]['img_points']
				draw_lines(dtm_rgb, lines, colors[angle])
			
			tile = filename.split('.')[0]
			res = tile + '.png'
			cv2.imwrite(os.path.join(dest_dir, res), dtm_rgb)

print('Processor Time: ', time.time() - start_time, 'seconds')

# python3.5 lineaments_merge.py