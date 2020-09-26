import numpy as np
import cv2
import os
from tqdm import tqdm

def compute_iou(targets, predictions):
	intersections = np.logical_and(targets, predictions)
	unions = np.logical_or(targets, predictions)
	iou_score = np.sum(intersections) / np.sum(unions)
	return iou_score

def get_road(true_masks, pred_masks):
	# class = road
	targets = []
	predictions = []
	for image_name in tqdm(sorted(os.listdir(true_masks))):
		img = cv2.imread(os.path.join(pred_masks, image_name),0)	
		img[img==0] = 0	
		img[img==255]=1 # road - white
		predictions.append(img)
		'''
		img_2 = cv2.imread(os.path.join(true_masks, image_name),0)
		img_2[img_2==30]=0 # background
		img_2[img_2==215]=1
		img_2[img_2==226]=1
		'''
		img_2 = cv2.imread(os.path.join(true_masks, image_name),0)
		img_2[img_2==30]= 0 # background
		img_2[img_2==215]= 1 # road - white
		img_2[img_2==226]= 0 #background
		targets.append(img_2)	
	return compute_iou(targets, predictions) * 100 # get percentage

def get_bg(true_masks, pred_masks):
	targets = []
	predictions = []
	for image_name in tqdm(sorted(os.listdir(true_masks))):
		img = cv2.imread(os.path.join(pred_masks, image_name),0)	
		img[img==0]=1	# bg - black to white
		img[img==255]=0
		predictions.append(img)

		img_2 = cv2.imread(os.path.join(true_masks, image_name),0)	
		img_2[img_2==30]= 1 # background 
		img_2[img_2==215]=0
		img_2[img_2==226]= 1 # background
		targets.append(img_2)	
	return compute_iou(targets, predictions) * 100 # get percentage



