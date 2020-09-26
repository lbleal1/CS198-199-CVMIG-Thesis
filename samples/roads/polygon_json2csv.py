'''
Given:
	json file of polygons

Do: Extract the width and height in a csv file
'''

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

import json 


import os
from tqdm import tqdm
import numpy
import pandas as pd

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
                regions.append(coords)
                
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

dataset_type = "train"
src_img = "assets/in_use/pad_1024/" +  dataset_type + "/"
csv_name = "1024Pad_OrigSat_train"

# convert json to dictionary
polycoords = Viajson2Polytuples(src_img + "via_export_" + dataset_type + "_aug.json")

#print(polycoords)
#print(len(polycoords))

#print(polycoords['tile_202.png'][0])
#first [0] - nth polygon in an image
#second [0] - nth tuple in a polygon
#third [0] - x coordinate ; [1] - y coordinate

width = []
height = []

for key in tqdm(list(polycoords.keys())):
	for pgon in range(len(polycoords[key])):
		x = []
		y = []
		for tup in range(len(polycoords[key][pgon])):
			x.append(polycoords[key][pgon][tup][0])
			y.append(polycoords[key][pgon][tup][1])

		width.append(max(x) - min(x))
		height.append(max(y) - min(y))

bboxes_dim = pd.DataFrame({'width': width, 'height': height})
bboxes_dim.to_csv('assets/csv/' + csv_name + '.csv')

