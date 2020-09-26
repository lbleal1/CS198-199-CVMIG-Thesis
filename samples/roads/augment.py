import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

import json 


import os
from tqdm import tqdm
import numpy
#numpy.random.bit_generator = numpy.random._bit_generator


def Viajson2Polytuples(json_file):
    polycoords = {}
    
    f = open(json_file)
    data = json.load(f)
    
    for k, v in data.items():
        regions = []
        if data[k]['regions']: # if there's a region
            for region in range(len(data[k]['regions'])):
                # extract x,y values
                x_vals = data[k]['regions'][region]['shape_attributes']['all_points_x']
                y_vals = data[k]['regions'][region]['shape_attributes']['all_points_y']
                #print("region")
                #print("x:", x_vals)
                #print("y:", y_vals)
        
                # create as a list of tuples
                coords = [ (x_vals[coord], y_vals[coord]) for coord in range(len(x_vals)) ]
                regions.append(Polygon(coords))
                
            polycoords.update( {v['filename']: regions})
        #else: 
            
    return polycoords

def getShapeAttrib(psoi_aug,poly_num):
    attrib_dict = {}
    
    attrib_dict.update( { "name": "polygon"} )
    
    all_points_x = []
    all_points_y = []
    

    for point_num in range(len(psoi_aug.polygons[poly_num].exterior)):
        all_points_x.append(int(psoi_aug.polygons[poly_num][point_num][0]))
        all_points_y.append(int(psoi_aug.polygons[poly_num][point_num][1]))
    
    attrib_dict.update( { "all_points_x": all_points_x } )
    attrib_dict.update( { "all_points_y": all_points_y } )
    
    
    
    return attrib_dict

def getRegions(psoi_aug):
    regions = []
    
    for poly_num in range(len(psoi_aug.polygons)): # how many regions
        new_dict = {}
        
        polys = getShapeAttrib(psoi_aug, poly_num)
        new_dict.update( { "shape_attributes" : polys } )
        
        new_dict.update( { "region_attributes" : {}   } )
        regions.append(new_dict)
    return regions

dataset_type = "valid"
src_img = "assets/in_use/default/aug_roads_test_split/" +  dataset_type + "/"
dst_augimg = "assets/in_use/resize/1024/"  +  dataset_type + "/"
dst_newjson = "assets/in_use/resize/1024/" +  dataset_type + "/"

# convert json to dictionary
polycoords = Viajson2Polytuples(src_img + "via_export_" + dataset_type + ".json")

'''
# define transforms
aug1 = iaa.Sequential([
    iaa.Affine(
        rotate = (-90)
    )
])

aug2 = iaa.Sequential([
    iaa.Affine(
        rotate = (90)
    )
])

aug3 = iaa.Sequential([
    iaa.Affine(
        rotate = (-180)
    )
])


aug4 = iaa.Sequential([
    iaa.Fliplr(1.0)
])

aug5 = iaa.Sequential([
    iaa.Fliplr(1.0),
    iaa.Affine(
        rotate = (-90)
    )
])

aug6 = iaa.Sequential([
    iaa.Fliplr(1.0),
    iaa.Affine(
        rotate = (90)
    )
])

aug7 = iaa.Sequential([
    iaa.Flipud(1.0)
])



aug = [aug1, aug2, aug3, aug4, aug5, aug6, aug7]
'''

#aug1 = iaa.Resize({"height": 1024, "width": 1024})

aug1 = iaa.Sequential([
    iaa.Resize({"height": 1024, "width": 1024})
    #iaa.PadToFixedSize(width=1024, height=1024, position="center")
])
aug = [aug1]

new_json = {}

for aug_num in range(len(aug)):
    transform_type = str(aug_num)
    for fname in tqdm(sorted(os.listdir(src_img))):
        inner_dict = {}
        if fname[-3:] == 'png': # json files also in the folder
            # read image
            img = imageio.imread(src_img + fname)

            if fname in polycoords:
                psoi = polycoords.get(fname) # get corresponding polygons   
                # for augment image + polygons
                psoi = ia.PolygonsOnImage(psoi,
                              shape=img.shape)

                image_aug, psoi_aug = aug[aug_num](image=img, polygons=psoi)
                #ia.imshow(psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7))

                # save image
                imageio.imwrite(dst_augimg + fname[:-4] + "__" + transform_type + ".png" , image_aug)

                # save json
                fsize = os.path.getsize(src_img + fname)
                inner_dict.update( {"filename": fname[:-4] + "__" + transform_type + ".png"} )
                inner_dict.update( {"size": fsize} )

                regions = getRegions(psoi_aug)
                inner_dict.update( {"regions": regions} )

                inner_dict.update( {"file_attributes": {}})
                keyname = fname[:-4] + "__" + transform_type + ".png" + str(fsize)
                new_json.update( { keyname : inner_dict})
                '''
                print()
                print(fname)
                print(psoi_aug)
                print()
                print(len(psoi_aug.polygons))
                print()
                print()
                #print(psoi_aug.polygons[0])
                #print()
                #print(psoi_aug.polygons[1])
                #print(len(psoi_aug.polygons[1].exterior)) #len of points
                #print(((int(psoi_aug.polygons[0][0][0])),int(psoi_aug.polygons[0][0][1])))
                #print()
                '''
            else: # no polygons
                # augment image and save
                psoi = Polygon([ (175,0), (214,1), (2,2), (1,1)])                
                psoi = ia.PolygonsOnImage([psoi],
                              shape=img.shape)
                image_aug, psoi_aug = aug[aug_num](image=img, polygons=psoi)
                imageio.imwrite( dst_augimg + fname[:-4] + "__" + transform_type + ".png" , image_aug)
                #ia.imshow(image_aug)
                # update json
                fsize = os.path.getsize(src_img + fname)
                inner_dict.update( {"filename": fname[:-4] + "__" + transform_type + ".png"} )
                inner_dict.update( {"size": fsize} )
                inner_dict.update( {"regions": []})
                inner_dict.update( {"file_attributes": {}})
                keyname = fname[:-4] + "__" + transform_type + ".png" + str(fsize)
                new_json.update( { keyname : inner_dict})
#print(new_json)
with open(dst_newjson + 'via_export_'+ dataset_type+ '_aug.json', 'w') as outfile:
    json.dump(new_json, outfile)

# merge old data to new data

## transfer old images
for fname in sorted(os.listdir(src_img)):
    if fname[-3:] == 'png':
        img = imageio.imread(src_img + fname)
        imageio.imwrite(dst_augimg+fname,img)


## merge json
# open old
with open(src_img + "via_export_"+ dataset_type + ".json") as json_file:
    d1 = json.load(json_file)
# open json for augmented data
with open(dst_newjson + 'via_export_'+ dataset_type+ '_aug.json') as json_file:
    d2 = json.load(json_file)
# append json for augmented data
for k,v in d2.items():
    d1.update( {k:v} )
# save
with open(dst_newjson + 'via_export_' +  dataset_type + '.json', 'w') as outfile:
    json.dump(d1, outfile)
    

