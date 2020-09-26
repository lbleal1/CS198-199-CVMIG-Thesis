# (due to cropping)
# do not save images whose
# mask is all black or only few roads 
# img consists of 50% of greater than white

import cv2
import numpy as np
import os
from tqdm import tqdm

src = "data"
dst = "data_cleaned"

src_imgs = [fname for fname in sorted(os.listdir(src+"/imgs"))]
src_masks = [fname for fname in sorted(os.listdir(src + "/masks"))]

outfname = "tile_"

for i in tqdm(range(len(src_imgs))):
  img_test = cv2.imread(src+"/imgs/" + src_imgs[i],  cv2.IMREAD_GRAYSCALE)
  mask_test = cv2.imread(src+"/masks/" + src_masks[i],  cv2.IMREAD_GRAYSCALE)

  if( (len((np.where(img_test==255))[1]) > (256*256*0.5)) or (len((np.where(mask_test==255))[1]) < (256*256*0.025)) ):
    continue
  else:
    cv2.imwrite(dst + "/imgs/" + outfname + str(i) + ".tif",cv2.imread(src+"/imgs/" + src_imgs[i]))
    cv2.imwrite(dst + "/masks/" + outfname + str(i) + "_mask.tif",  cv2.imread(src+"/masks/" + src_masks[i]))