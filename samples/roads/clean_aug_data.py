#cleans augmented data since in augmentation,
#the original data were also added

import os
import cv2

ds_type = "test"
src_dir = "./assets/in_use/resize/1024/"

for fname in sorted(os.listdir(src_dir + ds_type)):
	if fname[-4:] == ".png":
		img = cv2.imread(src_dir + ds_type + "/" + fname)
		if img.shape[0] <= 256 or img.shape[1] <= 256:
			print(fname)
			os.system("rm -f " + src_dir + ds_type + "/" + fname) 


