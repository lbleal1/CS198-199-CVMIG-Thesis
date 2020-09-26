import os
import sys

for i in range(1,10001):
  print( str(i) + ": "+ str(10001) )
  com_string_1 = "rsync -v -e ssh riza@202.92.132.222:\"/media/riza/'Seagate Bac'/road_building_extraction/final/segmentation/maskrcnn-matterport/Mask_RCNN/samples/roads/assets/output_data/ifsar/tiled_ifsar_png/tile_\""+ str(i) + ".png" + " "  +"/home/lois/Desktop/habagat/matterport/data/ifsar/unsplitted"
  os.system(com_string_1)

