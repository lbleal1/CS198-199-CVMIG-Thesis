#import json2mask
import fillin_test_masked
'''
src_dir = "samples/roads/assets/in_use/aug_roads_test_split"
dst_dir = "create_bimask/bi_violet/train_aug"
ds_type = "train"
json2mask.json2mask(src_dir, dst_dir, ds_type)
'''
# - fill in with blanks the image gaps in prediction and true masks
src_dir = "assets/in_use/aug_roads_test_split/valid" # use for masterlist of names
dst_masked = "create_bimask/bi_violet/valid/"
#dst_pred = "assets/results/scale_jitter/512/448/masks/"
fillin_test_masked.fill_in(src_dir, dst_masked)
#import fillin_test_masked
