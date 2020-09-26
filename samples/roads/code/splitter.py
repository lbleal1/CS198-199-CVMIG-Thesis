import os
import cv2
from random import Random
import math
from tqdm import tqdm


def splitter(src, dst):
	fnames = [ fname for fname in sorted(os.listdir(src))]

	Random(4).shuffle(fnames)

	train_per = 0.7
	val_per = 0.2
	test_per = 0.1

	train = []
	valid = []
	test = []

	for fname in tqdm(fnames[:math.ceil((train_per*len(fnames)))]):
		cv2.imwrite(dst + "train/" + fname,cv2.imread(src + fname)) 
		train.append(fname)
	for fname in tqdm(fnames[math.ceil(train_per*len(fnames)):math.ceil(train_per*len(fnames) + val_per*len(fnames))]):
		cv2.imwrite(dst + "valid/" + fname,cv2.imread(src + fname)) 
		valid.append(fname)
	for fname in tqdm(fnames[math.ceil(train_per*len(fnames) + val_per*len(fnames)):]):
		cv2.imwrite(dst + "test/" + fname,cv2.imread(src + fname))  
		test.append(fname)

	print(len(train))
	print(len(valid))
	print(len(test))

#src = "../assets/output_data/hard_lidar/split_prep/unsplitted/"

#dst = "../assets/output_data/hard_lidar/split_prep/splitted/"



