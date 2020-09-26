import os
os.chdir("../roads")

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

def create_blank_tm(image_name,dst_test_masked): # test_masked
	img = np.full((256,256), 30) # violet background
	im = Image.fromarray((img * 255).astype(np.uint8))
	im.save(dst_test_masked + image_name,"PNG")
	#cv2.imwrite(test_masked + image_name, img)
	print("Blank Test Mask Created for", image_name)

def create_blank_pred(image_name, dst_pred): # predictions
	img = np.full((256,256), 0) #black background
	im = Image.fromarray((img * 255).astype(np.uint8))
	im.save(dst_pred + image_name,"PNG")
	#cv2.imwrite(pred + image_name, img)
	print("Blank Pred Mask Created for", image_name)

def fill_in(test_dir, dst_test_masked, dst_pred):
	# set up images names
	test_images = []
	for image_name in os.listdir(test_dir):
		if image_name[-3:]=="png": #since there are json files also
			test_images.append(image_name)
		
	pred_images = [ image_name for image_name in os.listdir(dst_pred)]
	testmasked_images = [ image_name for image_name in os.listdir(dst_test_masked)]

	print("pred_images=", len(pred_images))
	print("test_images=", len(test_images))

	# fill in
	test_pred = list(set(test_images) - set(pred_images))
	print(test_pred)
	for image_name in tqdm(test_pred):
		create_blank_pred(image_name, dst_pred)

	test_testmasked = list(set(test_images) - set(testmasked_images))
	for image_name in tqdm(test_testmasked):
		create_blank_tm(image_name, dst_test_masked)

