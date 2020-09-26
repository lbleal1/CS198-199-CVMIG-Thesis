from PIL import Image
import numpy
from blend_modes import hard_light
from tqdm import tqdm
import os

bg_folder = "../assets/in_use/extra/lidar/lidar_roads_split" 
fg_folder = "../assets/in_use/default/roads_test_split"
dst_1 = "../assets/in_use/temp_roads_test_split"

dst_folder = "../assets/in_use/contrast/contrast_25_def"

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)



def combine_hard_light(bg_folder, fg_folder, dst_1):
    bg_names = [img for img in sorted(os.listdir(bg_folder)) if img[-4:]=='.png']
    fg_names = [img for img in sorted(os.listdir(fg_folder)) if img[-4:]=='.png']
    
    if bg_names == fg_names:
        print("Equal")
        for i in tqdm(range(len(bg_names))):
            # Import background image
            # bg is the hillshade
            background_img_raw = Image.open(bg_folder + "/" + bg_names[i]).convert('RGBA') # RGBA image
            background_img = numpy.array(background_img_raw) # Inputs to blend_modes need to be˓→numpy arrays.
            background_img_float = background_img.astype(float) # Inputs to blend_modes need to ˓→be floats.
            
            # Import foreground image 
            # fg is gsat
            foreground_img_raw = Image.open(fg_folder + "/" + fg_names[i]).convert('RGBA') # RGBA image
            foreground_img = numpy.array(foreground_img_raw) # Inputs to blend_modes need to be˓→numpy arrays.
            foreground_img_float = foreground_img.astype(float) # Inputs to blend_modes need to ˓→be floats.
            
            # Blend images
            opacity = 1 # The opacity of the foreground that is blended onto the background is ˓→70 %.
            blended_img_float = hard_light(background_img_float, foreground_img_float, opacity)
            
            # Convert blended image back into PIL image
            blended_img = numpy.uint8(blended_img_float) # Image needs to be converted back to ˓→uint8 type for PIL handling.
            blended_img_raw = Image.fromarray(blended_img) # Note that alpha channels are ˓→displayed in black by PIL by default.
            # This behavior is difficult to ˓→change (although possible).
            # If you have alpha channels in your ˓→images, then you should give
            # OpenCV a try.
            # Display blended image
            #blended_img_raw.show()

            # dst_1 for contrast
            blended_img_raw.save(dst_1 + '/' + bg_names[i])
            
    else:
        print("not equal")


folders = ["/train", "/test", "/valid"]

for folder in tqdm(folders):
    combine_hard_light(bg_folder + folder, fg_folder + folder, dst_1 + folder)


for folder in tqdm(folders):
    for img in tqdm(sorted(os.listdir(dst_1 + folder))):
        change_contrast(Image.open(dst_1 + folder + "/" + img), 25).save(dst_folder + folder + "/" + img)




