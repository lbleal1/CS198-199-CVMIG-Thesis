import numpy as np
import cv2
import os
from tqdm import tqdm

from metrics import iou
from metrics import dice

true_masks = "assets/in_use/diff_places/test_masked/scotabato"
pred_masks = "assets/results/diff_places/scotabato/masks/"


print("IoU:")
road_iou = iou.get_road(true_masks, pred_masks)
bg_iou = iou.get_bg(true_masks, pred_masks)
print("road iou=", road_iou)
print("background iou=", bg_iou)
print("mIoU =", (road_iou + bg_iou)/2)  # get percentage

print("\nDice:")
road_dice = dice.get_road(true_masks, pred_masks)
bg_dice = dice.get_bg(true_masks, pred_masks)
print("road dice=", road_dice)
print("background dice=", bg_dice)
print("dice =", (road_dice + bg_dice)/2) # get percentage

