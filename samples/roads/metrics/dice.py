import numpy as np
import cv2
import os
from tqdm import tqdm

def compute_dice(targets, predictions):
	intersections = np.logical_and(targets, predictions)
	total_pixels = np.sum(targets) + np.sum(predictions)
	dice_score = (2*np.sum(intersections))/total_pixels
	return dice_score

def get_road(true_masks, pred_masks):
	targets = []
	predictions = []
	for image_name in tqdm(sorted(os.listdir(true_masks))):
		img = cv2.imread(os.path.join(pred_masks, image_name),0)	
		img[img==0]=0	
		img[img==255]=1 # road 
		predictions.append(img)

		img_2 = cv2.imread(os.path.join(true_masks, image_name),0)
		img_2[img_2==30]=0 # background
		img_2[img_2==215]=1 # road
		img_2[img_2==226]=0 # background
		targets.append(img_2)	
	return compute_dice(targets, predictions) * 100 # get percentage


def get_bg(true_masks, pred_masks):
	targets = []
	predictions = []
	for image_name in tqdm(sorted(os.listdir(true_masks))):
		img = cv2.imread(os.path.join(pred_masks, image_name),0)	
		img[img==0]=1 # background
		img[img==255]=0
		predictions.append(img)

		img_2 = cv2.imread(os.path.join(true_masks, image_name),0)	
		img_2[img_2==30]=1 # background
		img_2[img_2==215]=0
		img_2[img_2==226]=1 # background
		targets.append(img_2)	
	return compute_dice(targets, predictions) * 100 # get percentage





