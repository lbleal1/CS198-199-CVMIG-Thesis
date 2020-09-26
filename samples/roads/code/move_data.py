import os
import sys

for i in range(1,10001):
  print( str(i) + ": "+ str(10001) )
  com_string = "mv ../assets/output_data/hard_lidar/tiled_hl_png/tile_" + str(i) + ".png" + " ../assets/output_data/hard_lidar/split_prep/unsplitted" 
  os.system(com_string)

