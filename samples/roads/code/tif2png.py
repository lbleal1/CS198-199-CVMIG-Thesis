import os
import cv2
from tqdm import tqdm

src_folder = "../assets/output_data/lidar/tiled_lidar/"
dst_folder = "../assets/output_data/lidar/tiled_lidar_png/"

def tif2png(src_folder, dst_folder):
	counter = 1
	for fname in tqdm(sorted(os.listdir(src_folder))):
		img = cv2.imread(src_folder + fname)
		cv2.imwrite(dst_folder + "tile_" + str(counter) + ".png", img)
		counter+=1

tif2png(src_folder, dst_folder)
