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
 

def convert_geo(dtm, points):
	res = []
	T0 = dtm.transform
	T1 = T0 * Affine.translation(0.5, 0.5)            
	for point in points:
		row1, col1, row2, col2 = point
		x1y1 = (row1, col1) * T1
		x2y2 = (row2, col2) * T1
		res.append(x1y1 + x2y2)
	return res

def otsu(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	high_thresh, thresh_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	low_thresh = 0.5 * high_thresh
	return (low_thresh, high_thresh)

def med(img, sigma=0.33):
	v = np.median(img)
	low_thresh = int(max(0, (1.0 - sigma) * v))
	high_thresh = int(min(255, (1.0 + sigma) * v))
	return (low_thresh, high_thresh)

start_time = time.time()

src_base = 'Hillshaded'
dest_base = 'Lineaments'
df_final = pd.DataFrame()
azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

for angle in azimuths:
	print('Processing: ', angle)
	src_dir = src_base + '/' + str(angle)
	dest_dir = dest_base + '/' + str(angle)
	file_list = glob.glob(os.path.join(src_dir, '*.tif'))

	for file in file_list:
		with rio.open(file) as dtm:
			dtm_arr = dtm.read(1, masked=True)
			dtm_rgb = cv2.cvtColor(dtm_arr, cv2.COLOR_GRAY2RGB)
			num = file.split('/')[-1].split('.')[0]
			dim = (dtm.width * dtm.height) // 2
			if np.ma.count_masked(dtm_arr) < dim:
				dtm_arr.filled(0)

				# res = 'hist_' + str(ctr) + '.png'
				# fig = plt.hist(np.uint8(dtm_arr).ravel(),256,[0,256])
				# res = 'orig_' + str(ctr) + '.png'
				# plt.imshow(np.uint8(dtm_arr), cmap="gray")
				# plt.savefig(res)
				# plt.clf()

				low_thresh, high_thresh = med(np.uint8(dtm_arr))
				edges = cv2.Canny(np.uint8(dtm_arr), low_thresh, high_thresh, None, 3)
				# edges = cv2.dilate(edges, (5,5), 7)
				res = 'edges_' + str(num) + '.png'
				cv2.imwrite(os.path.join(dest_dir, res), edges)

				lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=50, maxLineGap=20)
				img_points = []

				# If there are lineaments extracted, draw and save to dataframe
				if lines is not None:
					for l in lines:
						x1, y1, x2, y2 = l[0]
						cv2.line(dtm_rgb, (x1, y1), (x2, y2), (0, 0, 128), 1)
						img_points.append((x1, y1, x2, y2))
					
					geo_points = convert_geo(dtm, img_points)
				
					df = pd.DataFrame()
					df['img_points'] = img_points
					df['geo_points'] = geo_points
					df['filename'] = file.split('/')[-1]
					df['azimuth'] = angle
					df = df.reindex(columns=['filename', 'azimuth']+list(df.columns[0:2]))
					df_final = df_final.append(df)
				
				res = 'lines_' + str(num) + '.png'
				cv2.imwrite(os.path.join(dest_dir, res), dtm_rgb)
    
df_final = df_final.reset_index(drop=True)
df_final.to_csv(os.path.join(dest_base, 'lineaments_data.csv'), encoding='utf-8')

print('Processor Time: ', time.time() - start_time, 'seconds')

# python3.5 lineaments.py