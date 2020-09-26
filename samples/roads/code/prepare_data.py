# done after tiling via dem_tiler.py
# this prepares to replicate the exact same split of the existing roads_test_split data

# process
# 1. get 10,000 data to 10000_unsplitted
# 2. 10000_unsplitted tp splitter_1
# 3. splitter_1's valid folder to splitter_2

from splitter import splitter
import os
import sys
from tqdm import tqdm

root_dir = "../assets/output_data/lidar/"

def move_data(src, dst):
  for i in tqdm(range(1,10001)):
    os.system("mv " + src + "tile_" + str(i) + ".png" + " " + dst)

src_0 = root_dir + "tiled_lidar_png/"
dst_0 = root_dir + "10000_unsplitted"
move_data(src_0, dst_0)

src_1 = "../assets/output_data/lidar/10000_unsplitted/"
dst_1 =  "../assets/output_data/lidar/splitted_1/"
splitter(src_1, dst_1)

src_2 = "../assets/output_data/lidar/splitted_1/valid/"
dst_2 = "../assets/in_use/lidar_roads_split/"
splitter(src_2, dst_2)
