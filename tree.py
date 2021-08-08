import rasterio
from pyproj import Transformer
import numpy as np
import os


def addTrees(elevation, centerlon, centerlat):

    baseRasterPath = 'c:/art/raster'
    numTrees = 1000

    treeFile = rasterio.open(os.path.join(baseRasterPath, 'tree.tif'))
    tree = treeFile.read(1)
    print ('raster size')
    print (tree.shape)

    # Get the cell size of the raster
    cellx, celly  = treeFile.res
    cellxHalf, cellyHalf = cellx/2.0, celly/2.0


    linear_idx = np.random.choice(tree.size,
                                  size = numTrees, replace=False,
                                  p=tree.ravel()/float(tree.sum()))
    x, y = np.unravel_index(linear_idx, tree.shape)
    transformer = Transformer.from_crs(treeFile.crs, "epsg:4326", always_xy=True)
    cx, cy = rasterio.transform.xy(treeFile.transform, x, y, offset='center')
    cx = np.array(cx) + np.random.random(len(cx)) * cellx - cellxHalf
    cy = np.array(cy) + np.random.random(len(cy)) * celly - cellyHalf

    coords = [transformer.transform(x, y) for x,y in zip(cx, cy)]
    for lon, lat in coords:
        print (lon, lat)



    return []

