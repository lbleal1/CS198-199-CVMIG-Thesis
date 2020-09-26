import pandas as pd

import os
import gdal
from datetime import datetime
import argparse
import shutil
import math


import glob
import numpy as np
import torch

from utils import data_utils
from utils import augmentation as aug
from utils import metrics
from models import unet

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

#%matplotlib inline

# open tensorboard
# tensorboard --logdir='./logs' --port=6006
import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm


import torch
import torch.optim as optim
import time
import argparse
import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import torch
import torchvision.transforms.functional as F
from PIL import Image

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS
import itertools

def tiler( in_tifname, ftype, tile_size):
	in_path = "../../assets/raw_data/" 
    
	if ftype == "lidar":
		output_folder = "../../assets/output_data/tiled_raw_lidar/"
		band_number = 1
	elif ftype == "gmaps":
		output_folder = "../../assets/output_data/tiled_gmaps/"
		band_number = 3
    
	output_filename = 'tile_'
	
	ds = gdal.Open(in_path + in_tifname)
	band = ds.GetRasterBand(band_number)
	xsize = band.XSize
	ysize = band.YSize

	#xsize = 5000
	#ysize = 5000

	tile_size_y = 0
	tile_size_x = 0

	counter = 0
	while tile_size_x < xsize:
		#print("tile_size_x= " + str(tile_size_x) + "/" + str(xsize))
		#print("tile_size_y= " + str(tile_size_y) + "/" + str(ysize))	
		if tile_size_y > ysize:
			tile_size_y = 0
			tile_size_x += tile_size
		com_string = "gdal_translate -of GTIFF -srcwin " + str(tile_size_x)+ " " + str(tile_size_y) + " " + str(tile_size) + " " + str(tile_size) + " \"" + str(in_path) + str(in_tifname) + "\" " + "\""+ str(output_folder) + str(output_filename) + "\"" + str(tile_size_x) + "_" + str(tile_size_y) + ".tif"
		os.system(com_string)  
		tile_size_y += tile_size
		print("tiling = " + str(counter))
		counter +=1
		if(tile_size_x > xsize):
			print("bruh I'm done")
	print("Done Tiling")

# gdal_translate -b 1 -b 2 -b 3 input.tif output.tif

def toThreeChannels(in_folder, out_folder):
    src_folder ='../../assets/output_data/' + in_folder + "/"
    dst_folder = '../../assets/output_data/' + out_folder + "/"
    img_src = []
    img_dst = []
    for fname in sorted(os.listdir(src_folder)):
        print(fname)
        img_src.append(os.path.join(src_folder, fname))
        img_dst.append(os.path.join(dst_folder, fname))

    for i in range(len(img_src)):
        com_string = "gdal_translate -b 1 -b 2 -b 3 " + img_src[i] + " " + img_dst[i] 
        os.system(com_string)
        print(i)

def files2csv(in_folder, out_csv):
    in_path = "../../assets/output_data/" + in_folder + "/**/*.tif"
    
    get_fnames = glob.glob(in_path, recursive=True)
    
    df = pd.DataFrame(get_fnames)
    
    df.rename(columns={0:'file_path'}, inplace=True)
    df['sat_img_path'] = df['file_path'].apply(lambda x: x.split('/')[-1])
    df['map_img_path'] = df['file_path'].apply(lambda x: x.split('/')[-1])
    df['sat_map'] = df['file_path'].apply(lambda x: x.split('/')[-2])
    df['train_valid_test'] = df['file_path'].apply(lambda x: x.split('/')[-3])
    
    df.sort_values(by=['file_path'], inplace=True)
    
    outpath_csv = "../../assets/output_data/csv_files/" + out_csv
    df.to_csv(outpath_csv, index=False)
    
    return outpath_csv, out_csv


def adf2tif(adf_fname, tif_fname):
    adf_path = "../../assets/raw_data/"
    tif_path = "../../assets/output_data/"
    os.system("gdal_translate " + adf_path + adf_fname + " " + tif_path + tif_fname )

def segmenter(run, data_num, in_folder, out_csv, start, stop):
    #outpath_csv, outcsv = files2csv(in_folder, out_csv)
    outpath_csv = "../../assets/output_data/csv_files/" + out_csv       

    model = unet.UNetSmall()
    model.eval() # setting to eval
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        print("Cuda is available.")
        model = model.cuda()
   
    checkpoint = torch.load('../checkpoints/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
   
    mass_dataset_test = data_utils.MassRoadBuildingDataset( run, data_num, outpath_csv , start, stop, in_folder,  'test',  transform=transforms.Compose([aug.ToTensorTarget()]))
    test_dataloader = DataLoader(mass_dataset_test, batch_size=1, num_workers=3, shuffle=False)  
 
    seg_res = []
    sat_imgs = []
    
    segres_unrolled = []
    sat_unrolled = []

    counter = 0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader, desc='test' + " run= " + str(run) )):
            # get the inputs and wrap in Variable
            if torch.cuda.is_available():
                inputs = Variable(data['sat_img'].cuda(), volatile=True)
                labels = Variable(data['sat_img'].cuda(), volatile=True)
            else:
                inputs = Variable(data['sat_img'], volatile=True)
                labels = Variable(data['sat_img'], volatile=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.nn.functional.sigmoid(outputs)
            seg_res.append(outputs)
            sat_imgs.append(inputs)
            counter+=1

    # unroll segmentation output
    segres_unrolled = [ np.squeeze((seg_res[i].data).cpu().numpy()) for i in range(len(seg_res))]

    # unroll sat_imgs
    sat_unrolled = [ np.squeeze((sat_imgs[i].data).cpu().numpy()) for i in range(len(sat_imgs))]
    
    return segres_unrolled, sat_unrolled

def eraser(seg_result, raw_lidar_unrolled,run):
    # cleaning segres
    erased_lidar = [ np.zeros((256,256)) for i in range(len(seg_result))]

    for i in range(len(seg_result)):
        print("erasing=" + str(i) + " run= " + str(run))
        for j in range(256):
            for k in range(256):
                if(seg_result[i][j][k] > 0.04):
                    erased_lidar[i][j][k] = 0
                else:
                    erased_lidar[i][j][k] = raw_lidar_unrolled[i][j][k]
    return erased_lidar

def pixel2coord(col, row, a, b, c, d, e, f):
    """Returns global coordinates to pixel center using base-0 raster index"""
    xp = (a) * col + b * row + c 
    yp = (d) * col + e * row + f
    return(xp, yp)

def tiffer_saver(erased_lidar, raw_lidar_df, run):
    for i in range(len(erased_lidar)):
        print("tiffing and saving=" + str(i) + " run= " + str(run))
        array = erased_lidar[i]
        ds = gdal.Open(raw_lidar_df['file_path'][i])
        
        c, a, b, f, d, e = ds.GetGeoTransform()
        
        # initializations
        rows, colms = np.shape(array) # get columns and rows of your image from gdalinfo
        lat = np.empty(shape=(rows,colms))
        lat.fill(0)
        lon = np.empty(shape=(rows,colms))
        lon.fill(0)


        for row in  range(0,rows):
            for col in  range(0,colms): 
                lon[row][col], lat[row][col] = pixel2coord(col,row,a,b,c,d,e,f)
        
        xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
        nrows,ncols = np.shape(array)
        xres = (xmax-xmin)/float(ncols)
        yres = (ymax-ymin)/float(nrows)
        geotransform=(xmin,xres,0,ymax,0, -yres)   
        # That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
        #         top left y, rotation (0 if North is up), n-s pixel resolution)
        # I don't know why rotation is in twice???
        
        output_raster = gdal.GetDriverByName('GTiff').Create('../../assets/output_data/tiled_erased_lidar/'+raw_lidar_df['sat_img_path'][i],ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
        print(output_raster)
        output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
        srs = osr.SpatialReference()                 # Establish its coordinate encoding
        srs.ImportFromEPSG(32651)                     # This one specifies WGS84 lat long.
                                                    
                                                   
        output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system to the file
        output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster

        output_raster.FlushCache()

import cv2
import numpy as np

def seg_save(seg_result, df_paths, run):
    for i in range(len(seg_result)):
        print("saving segmentation output... " + str(i) + " run = " + str(run))
        if str(type(seg_result[i])) != "<class 'numpy.ndarray'>":
            cv2.imwrite("../../assets/output_data/seg_result/" + df_paths['sat_img_path'][i][:-3] + "png",np.array(seg_result[i]))
        else:
            cv2.imwrite("../../assets/output_data/seg_result/" + df_paths['sat_img_path'][i][:-3] + "png",seg_result[i])
        for row in  range(0,rows):
            for col in  range(0,colms): 
                lon[row][col], lat[row][col] = pixel2coord(col,row,a,b,c,d,e,f)
        
        xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
        nrows,ncols = np.shape(array)
        xres = (xmax-xmin)/float(ncols)
        yres = (ymax-ymin)/float(nrows)
        geotransform=(xmin,xres,0,ymax,0, -yres)   
        # That's (top left x, w-e pixel resolution, rotation (0 if North is up), 
        #         top left y, rotation (0 if North is up), n-s pixel resolution)
        # I don't know why rotation is in twice???

        output_raster = gdal.GetDriverByName('GTiff').Create('../../assets/output_data/tiled_erased_lidar/'+raw_lidar_df['sat_img_path'][i],ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
        output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
        srs = osr.SpatialReference()                 # Establish its coordinate encoding
        srs.ImportFromEPSG(32651)                     # This one specifies WGS84 lat long.
                                                    
                                                   
        output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system to the file
        output_raster.GetRasterBand(1).WriteArray(array)   # Writes my array to the raster

        output_raster.FlushCache()

# raw LiDAR dimensions: 46728, 86749


def get_diff(a, b):
	return a[0]-b[0], a[1]-b[1]

def get_dim(orig_tif):
    
    return 


def stitcher(in_folder, out_folder, tile_size, orig_tif):
    in_fp = "../../assets/output_data/" + in_folder
    out_fp = "../../assets/output_data/" + out_folder
    
    filepaths = glob.glob(os.path.join(in_fp, '*.tif'))
    filenames = [x.split('/')[-1].split('.')[0] for x in filepaths]
    
    raster = gdal.Open(orig_tif)
    
    rows = raster.RasterYSize #rows = 86749
    cols = raster.RasterXSize #cols = 46728
        
    print(rows, cols)
    new_cols = math.ceil(cols/tile_size)
    new_rows = math.ceil(rows/tile_size)

    for i in range(0, cols, tile_size):
        nums = list(range(i, i+tile_size, 256))
	
        for j in range(0, rows, tile_size):
            nums2 = list(range(j, j+tile_size, 256))
            labels = list(itertools.product(nums, nums2))
            fp_list = []

            for idx, k in enumerate(labels):
                name = 'tile_' + str(k[0]) + '_' + str(k[1])
                if name in filenames:
                    fp_list.append(in_fp + '/' + name + '.tif')
                    filenames.remove(name)
                else:
                    prev = labels[idx-1]
                    diff = get_diff(prev, k)

                    prev_name = 'tile_' + str(prev[0]) + '_' + str(prev[1])
                    prev_fp = in_fp + '/' + prev_name + '.tif'
                    prev_tile = rasterio.open(prev_fp)
                    
                    arr = np.zeros((256, 256), dtype="float32")
                    x,y = prev_tile.bounds.left + diff[0], prev_tile.bounds.top + diff[1]
                    new_transform = rasterio.transform.from_origin(x, y, 1, 1)
                    new_tile = name + '.tif'
                    new_meta = prev_tile.meta.copy()
                    new_meta.update({"driver": "GTiff",
                        "height": arr.shape[0],
                        "width": arr.shape[1],
                        "transform": new_transform,
                        "crs": CRS.from_epsg(32651),
                        })

                    with rasterio.open(os.path.join(in_fp, new_tile), "w", **new_meta) as dest:
                        dest.write(arr, 1)
                    # print("Writing: ", new_tile)
                    fp_list.append(in_fp + '/' + new_tile)
                    print("i =" + str(i) + " j =" + str(j))
            
            to_mosaic = []
            for fp in fp_list:
                src = rasterio.open(fp)
                to_mosaic.append(src)

            mosaic, out_trans = merge(to_mosaic)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": CRS.from_epsg(32651),
                })

            new_file = 'tile_' + str(i) + '_' + str(j) + '.tif'
            with rasterio.open(os.path.join(out_fp, new_file), "w", **out_meta) as dest:
                dest.write(mosaic)
            print("Saving: ", new_file)
            print(mosaic.shape)

def getAllFiles(in_fp):
    fp_list = []
    for fname in sorted(os.listdir(in_fp)):
        fp_list.append(in_fp + "/" + fname)
    return fp_list


def read_dict(dict_names, dictios):
    f = open("../../assets/output_data/runtimes.txt","w+")
    for i in range(len(dict_names)):
        print(dict_names[i])     
        f.write(dict_names[i])
        for k,v in dictios[i].items():
            print(k,v)
            f.write(str((k,v)))
    f.close()

