import os
import cv2
from tqdm import tqdm
src_dir = "bi_violet"
dst_dir = "masks"

folders = ["test"]

for folder in tqdm(folders):
  for img in tqdm(sorted(os.listdir(src_dir+ "/" + folder))):
    mask = cv2.imread(os.path.join(src_dir,folder,img),0)
    mask[mask==30] = 0
    mask[mask==215] = 255 # road - white
    mask[mask==226] = 0
    cv2.imwrite(os.path.join(dst_dir, folder, img),mask)
    
