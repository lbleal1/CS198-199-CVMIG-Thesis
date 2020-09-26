import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.roads import road

ROAD_DIR = os.path.join(ROOT_DIR, "samples/roads/assets/in_use/roads_test_split")

for i in ["train", "valid", "test"]:
	dataset = road.RoadDataset()
	dataset.load_road(ROAD_DIR, i)

	# Must call before using the dataset
	dataset.prepare()
	all_samp = len([img for img in os.listdir(ROAD_DIR+"/"+i)]) - 4 # since json annotation and project are included
	pos = len(dataset.image_ids)	
	neg = all_samp - pos
	print(i + " total images:" + str(all_samp))
	print("Positive Samples:", pos)
	print("positive %:", (pos/all_samp) * 100 )
	print("Negative Samples:",  neg)
	print("negative %:", (neg/all_samp) * 100 )
	print()
