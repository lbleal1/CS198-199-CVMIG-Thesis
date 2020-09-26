import os
os.chdir("../../roads")
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model
import mrcnn.model as modellib
from mrcnn.model import log
import cv2


from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

#place custom_1.py inside the folder where this ipynb is run
from samples.roads import road 

def json2mask(src_dir,dst_dir, ds_type):
	
	#src = "samples/roads/assets/in_use/aug_roads_test_split"
	#dst_dir = "assets/in_use/aug_roads_test_split/test_masked/test_piece_masked"

	## Ground Truth Data Generation
	# Load dataset
	dataset = road.RoadDataset()
	custom_DIR = os.path.join(ROOT_DIR, src_dir)
	#place your json file inside train folder
	dataset.load_road(custom_DIR, ds_type)
	# Must call before using the dataset
	dataset.prepare()

	print("Image Count: {}".format(len(dataset.image_ids)))
	print("Class Count: {}".format(dataset.num_classes))
	for i, info in enumerate(dataset.class_info):
		print("{:3}. {:50}".format(i, info['name']))


	## Process
	#image_ids = np.random.choice(dataset.image_ids, 60)
	os.chdir(dst_dir)
	index = dataset.image_ids
	#index=range(1,5)
	#print(image_ids)
	to_display = []
	titles = []
	for image_id in tqdm(index):
		image = dataset.load_image(image_id)
		mask, class_ids = dataset.load_mask(image_id)
		#mask=mask[:,:,-1]
		# Pick top prominent classes in this image
		unique_class_ids = np.unique(class_ids)
		#print(unique_class_ids)
		mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]]) for i in unique_class_ids]
		top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),key=lambda r: r[1], reverse=True) if v[1] > 0]
	    
		# Generate images and titles
		for i in range(1):
			class_id = top_ids[i] if i < len(top_ids) else -1
			# Pull masks of instances belonging to the same class.
			m = mask[:, :, np.where(class_ids == class_id)[0]]
			m = np.sum(m * np.arange(1, m.shape[-1] +1), -1)
			m[m>1]=1
			#print(np.unique(m))
			to_display.append(m)
			titles.append(dataset.class_names[class_id] if class_id != -1 else "-")
			mask = dataset.image_info[image_id]['id']
			#print(m.shape)
			plt.imsave(mask,m)
	
