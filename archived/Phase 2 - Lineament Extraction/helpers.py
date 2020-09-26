import os
import glob
import cv2
import numpy as np
import rasterio as rio
import pandas as pd
import phasepack as pp
from skimage.draw import line
from affine import Affine
from osgeo import gdal


def hillshade(src_dir, in_path, out_path):
	"""
		Function that produces shaded relief images from the raw DTM
	"""
	file_list = glob.glob(os.path.join(src_dir, '*.tif'))
	azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

	for file in file_list:
		filename = file.split('/')[-1].split('.')[0]

		# one-directional hillshading
		for angle in azimuths:
			dest_dir = out_path + str(angle)
			# if not os.path.exists(dest_dir):
			# 	os.makedirs(dest_dir)
			com_string = "gdaldem hillshade -compute_edges -az " + str(angle) + " " + str(in_path) + str(filename) + ".tif " + str(dest_dir) + "/" + str(filename) + ".tif"
			os.system(com_string)

		# multi-directional hillshading
		dest_dir = out_path + 'multi'
		# if not os.path.exists(dest_dir):
		# 	os.makedirs(dest_dir)
		com_string = "gdaldem hillshade -multidirectional -compute_edges " + str(in_path) + str(filename) + ".tif " + str(dest_dir) + "/" + str(filename) + ".tif"
		os.system(com_string)
	return

def convert_geo(dtm, points):
	"""
		Function that converts pixel coordinates to geographic coordinates
	"""
	res = []
	T0 = dtm.transform
	T1 = T0 * Affine.translation(0.5, 0.5)            
	for point in points:
		row1, col1, row2, col2 = point
		x1y1 = (row1, col1) * T1
		x2y2 = (row2, col2) * T1
		res.append(x1y1 + x2y2)
	return res

def canny_thresh(img, sigma=0.33):
	"""
		Function that computes for the upper and lower thresholds for canny edge detection
	"""
	v = np.median(img)
	low_thresh = int(max(0, (1.0 - sigma) * v))
	high_thresh = int(min(255, (1.0 + sigma) * v))
	return (low_thresh, high_thresh)

def phase_thresh(img, t=0.15):
	"""
		Function that binarizes the output of phase congruency / phase symmetry edge detection
	"""
	h = img.shape[0]
	w = img.shape[1]
	for y in range(0, h):
		for x in range(0, w):
			img[y, x] = 255 if img[y,x] > t else 0
	return img

def extract_lineaments(src_base, dest_base, smoothing, edge_method):
	"""
		Function that performs the main lineament extraction process
	"""
	df_final = pd.DataFrame()
	azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

	for angle in azimuths:
		print('Processing: ', angle)
		src_dir = src_base + '/' + str(angle)
		dest_dir = dest_base + '/' + str(angle)
		file_list = glob.glob(os.path.join(src_dir, '*.tif'))

		# if not os.path.exists(dest_dir):
		# 	os.makedirs(dest_dir)

		for file in file_list:
			with rio.open(file) as dtm:
				dtm_arr = dtm.read(1, masked=True)
				dtm_rgb = cv2.cvtColor(dtm_arr, cv2.COLOR_GRAY2RGB)
				num = file.split('/')[-1].split('.')[0]
				dim = (dtm.width * dtm.height) // 2

				if np.ma.count_masked(dtm_arr) < dim:
					dtm_arr.filled(0)

					if smoothing == True:
						dtm_arr = cv2.GaussianBlur(np.uint8(dtm_arr), (7,7), 0)
					
					if edge_method == 'canny':
						low_thresh, high_thresh = canny_thresh(dtm_arr)
						edges = cv2.Canny(dtm_arr, low_thresh, high_thresh, None, 3)
					elif edge_method == 'phase_cong':	
						pc, ori, ft, T = pp.phasecongmono(dtm_arr, k=15)
						edges = phase_thresh(pc)
					else:	#edge_method == 'phase_sym'
						ps, totalEnergy, Th = pp.phasesymmono(dtm_arr, k=15)
						edges = phase_thresh(ps)
										
					res = 'edges_' + str(num) + '.png'
					cv2.imwrite(os.path.join(dest_dir, res), edges)

					lines = cv2.HoughLinesP(np.uint8(edges), 1, np.pi/180, threshold=100, minLineLength=20, maxLineGap=40)
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
	return

def to_tuple(string, dtype):
	"""
		Function that parses data from .csv file input
	"""
	if dtype == 'int':
		return tuple(int(a) for a in string.strip('()').split(','))
	else:	# dtype == 'float'
		return tuple(float(a) for a in string.strip('()').split(','))

def draw_lines(img, lines, color):
	"""
		Helper function for map_lineaments, unpacks lineament endpoints and draws on image
	"""
	for line in lines:
		line = to_tuple(line, 'int')
		x1, y1, x2, y2 = line
		cv2.line(img, (x1, y1), (x2, y2), color, 1)
	return

def map_lineaments(csv_file, src_dir, dest_dir):
	"""
		Function that compiles all lineaments extracted per tile and maps these lineaments to the multi-directional hillshaded tile
	"""
	df = pd.read_csv(csv_file)
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

	# if not os.path.exists(dest_dir):
	# 	os.makedirs(dest_dir)
	for file in file_list:
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
				res = 'lines_' + tile + '.png'
				# print(res)
				cv2.imwrite(os.path.join(dest_dir, res), dtm_rgb)
	return

def for_plotting(csv_file, dest_dir, dest_file):
	"""
		Function that transforms the main .csv file of lineament data to a version that can be imported in QGIS
	"""
	df = pd.read_csv(csv_file)	
	# if not os.path.exists(dest_dir):
	# 	os.makedirs(dest_dir)

	lines = df['geo_points']
	lines_converted = []
	for line in lines:
	    line = to_tuple(line, 'float')
	    lines_converted.append(line)

	print('Number of lineaments: ', len(lines_converted))
	df_new = pd.DataFrame(lines_converted, columns=['x_from', 'y_from', 'x_to', 'y_to'])
	df_new.to_csv(os.path.join(dest_dir, dest_file))
	return