from PIL import Image
import numpy
from blend_modes import hard_light
from tqdm import tqdm
import os

dst_1 = "../assets/in_use/default/aug_roads_test_split"
dst_folder = "../assets/in_use/contrast/aug_contrast_25_def"
contrast_level = 15
folders = ["/train", "/test", "/valid"]

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

for folder in tqdm(folders):
    for img in tqdm(sorted(os.listdir(dst_1 + folder))):
        if img[-4:] == ".png":
            change_contrast(Image.open(dst_1 + folder + "/" + img), contrast_level).save(dst_folder + folder + "/" + img)
