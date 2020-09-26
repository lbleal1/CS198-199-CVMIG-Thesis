# do save 
# - complete and masks for the prediction
# - creates true masks form the test dataset
# - fill in with blanks the image gaps in prediction and true masks


from process_results import json2mask
from process_results import saving
from process_results import fillin_test_masked


# create true masks from the test dataset
test_dir = "samples/roads/assets/in_use/diff_places/scotabato"
dst_dir = "assets/in_use/diff_places/test_masked/scotabato"
json2mask.json2mask(test_dir, dst_dir)


# complete and masks for the prediction
weights_path = "logs/dataset/GoogleSat_only/contrast_15/aug/mask_rcnn_roads_1028.h5"
road_ds = "samples/roads/assets/in_use/diff_places/scotabato"
save_dirs = ["samples/roads/assets/results/diff_places/scotabato/masks", "samples/roads/assets/results/diff_places/scotabato/complete" ]
saving.save_pred(weights_path, road_ds, save_dirs)

# - fill in with blanks the image gaps in prediction and true masks
test_dir = "assets/in_use/diff_places/scotabato/test" # use for masterlist of name
dst_test_masked = "assets/in_use/diff_places/test_masked/scotabato/"
dst_pred = "assets/results/diff_places/scotabato/masks/"
fillin_test_masked.fill_in(test_dir, dst_test_masked, dst_pred)

