import os
import glob
from itertools import product
import rasterio as rio
from rasterio import windows
import numpy as np
import matplotlib.pyplot as plt
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import time

def write_file(dest, file, dtm_meta, hillshade):
    with rio.open(os.path.join(dest, os.path.basename(file)), 'w', **dtm_meta) as dst:
        dst.write(hillshade.astype(rio.float32), 1)


start_time = time.clock()
start_runtime = time.time()

azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
altitude = 45

data_dir = 'Tiled Ilocos'
dest_dir = 'Hillshaded'
file_list = glob.glob(os.path.join(data_dir, '*.tif'))

for file in file_list:
    with rio.open(file) as src:
        elevation = src.read(1)
        dim = (src.width * src.height) // 2
        # If masked values are less than half of the total num of pixels
        if np.sum(elevation < 0) < dim:
            # Set masked values to np.nan
            elevation[elevation < 0] = np.nan
            # Get metadata of TIFF file
            dtm_meta = src.meta
            dtm_meta.update(dtype=rio.float32)
            for x in azimuths:
                hillshade = es.hillshade(elevation, azimuth=x, altitude=altitude)

                dest = dest_dir + '/' + str(x)
                write_file(dest, file, dtm_meta, hillshade)

print('Processor Time: ', time.clock() - start_time, 'seconds')
print('Wall Time: ', time.time() - start_time, 'seconds')

# Checking output of one tile
for x in azimuths:
    filename = 'Hillshaded/' + str(x) + '/Tiled Ilocos.100.tif'
    with rio.open(filename) as src:
        hillshade = src.read(1)

    title = 'Azimuth = ' + str(x)
    ep.plot_bands(
        hillshade,
        scale=False,
        cbar=False,
        title=title,
        figsize=(5, 5),
    )
    png = 'hillshade_' + str(x) + '.png'
    plt.savefig(png)