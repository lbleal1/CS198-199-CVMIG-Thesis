import os
os.chdir("../roads")

import sys
import random
import math
import re
import time
import numpy as np
import skimage
from skimage import io
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from process_results import process_image

ROOT_DIR = os.path.abspath("../../") 

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.roads import road

def save_pred(weights_path, road_ds, save_dirs):
	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Path to trained weights
	ROAD_WEIGHTS_PATH =  os.path.join(ROOT_DIR, weights_path)
	ROAD_DIR = os.path.join(ROOT_DIR, road_ds)
	IMAGE_DIR = os.path.join(ROAD_DIR, "test")
	save_modes = [3, 0]
	save_dirs[0] = os.path.join(ROOT_DIR, save_dirs[0])
	save_dirs[1] = os.path.join(ROOT_DIR, save_dirs[1])	


	'''
	save_mode = 0 , save image with bbox,class_name,score and mask;
	save_mode = 1 , save image with bbox,class_name and score;
	save_mode = 2 , save image with class_name,score and mask;
	save_mode = 3 , save mask with black background;
	'''

	## Configurations
	config = road.RoadConfig()

	# Override the training configurations with a few
	# changes for inferencing.
	class InferenceConfig(config.__class__):
		# Run detection on one image at a time
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
		IMAGE_RESIZE_MODE = "square"
		IMAGE_MIN_DIM = 800		
		IMAGE_MAX_DIM = 1024

		RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
		MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
	config = InferenceConfig()
	config.display()


	## SETUP
	DEVICE = "/gpu:0"  
	TEST_MODE = "inference"

	## Load Test Dataset
	dataset = road.RoadDataset()
	dataset.load_road(ROAD_DIR, "test")
	dataset.prepare() # Must call before using the dataset
	print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

	## Load Model
	# Create model in inference mode
	with tf.device(DEVICE):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
		                      config=config)

	# Load weights
	print("Loading weights ", ROAD_WEIGHTS_PATH )
	model.load_weights(ROAD_WEIGHTS_PATH, by_name=True)

	# save image

	for image_name  in tqdm(sorted(os.listdir(IMAGE_DIR))):
		if image_name[-3:] == "png":    
			image = skimage.io.imread(os.path.join(IMAGE_DIR,image_name))
			if image.shape[-1] == 4:
				image = image[..., :3]
			results = model.detect([image], verbose=1)
			r = results[0]
			process_image.save_images(image, image_name, r['rois'], r['masks'],
		    r['class_ids'],r['scores'],dataset.class_names,save_dir = save_dirs[0], mode=save_modes[0])
			process_image.save_images(image, image_name, r['rois'], r['masks'],
		    r['class_ids'],r['scores'],dataset.class_names,save_dir = save_dirs[1], mode=save_modes[1])	

