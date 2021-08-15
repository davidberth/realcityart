import rasterio
from pyproj import Transformer
import numpy as np
import os
from transforms3d.affines import compose
import trimesh
import xarray as xr


def addTrees(elevation, centerlon, centerlat):

    meshes = []
    baseRasterPath = 'c:/art/raster'

    treeFile = rasterio.open(os.path.join(baseRasterPath, 'tree.tif'))
    tree = treeFile.read(1)
    sizex, sizey = tree.shape

    # Get the cell size of the raster
    cellx, celly  = treeFile.res
    cellxHalf, cellyHalf = cellx/2.0, celly/2.0


    x = []
    y = []
    for lx in range(sizex):
        for ly in range(sizey):
            numTreesInCell = tree[lx, ly]
            if numTreesInCell < 15:
                numTrees = 0
            elif numTreesInCell < 30:
                numTrees = 1
            elif numTreesInCell < 75:
                numTrees = 4
            else:
                numTrees = 9

            sqrtNumTrees = int(np.sqrt(numTrees) + 0.001)

            for subx in range(sqrtNumTrees):
                for suby in range(sqrtNumTrees):
                    x.append(lx + subx/sqrtNumTrees)
                    y.append(ly + suby/sqrtNumTrees)

    transformer = Transformer.from_crs(treeFile.crs, "epsg:4326", always_xy=True)
    cx, cy = rasterio.transform.xy(treeFile.transform, x, y, offset='center')
    cx = np.array(cx) + np.random.random(len(cx)) * cellx - cellxHalf
    cy = np.array(cy) + np.random.random(len(cy)) * celly - cellyHalf

    r = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rotation matrix
    z = [1.0, 1.0, 1.0]  # zooms

    coords = [transformer.transform(x, y) for x,y in zip(cx, cy)]
    lons, lats = zip(*coords)

    xx = xr.DataArray(list(lons), dims="points")
    yy = xr.DataArray(list(lats), dims="points")
    elevs = elevation.interp(x=xx, y=yy, method='nearest').values

    print ('adding trees')

    for lon, lat, elev in zip(lons, lats, elevs):

        if not np.isnan(elev):
            renderx = (lon - centerlon) * 111000.0
            rendery = (lat - centerlat) * 111000.0
            renderz = elev

            treeHeight = np.random.randint(7, 14)
            leavesRadius = np.random.randint(8, 14)
            t = [renderx, rendery, renderz * 2.0 + treeHeight/2.0]
            trans = compose(t,r,z)

            trunk = trimesh.creation.cylinder(2.0, treeHeight, transform = trans,
                                              sections=4)
            trunk.visual.mesh.visual.face_colors = [90, 55, 0, 200]
            meshes.append(trunk)

            leaves = trimesh.creation.icosphere(subdivisions=1, radius=leavesRadius)
            t = [renderx, rendery, renderz * 2.0 + treeHeight / 2.0 + leavesRadius]
            trans = compose(t, r, z)
            leaves.apply_transform(trans)
            leaves.visual.mesh.visual.face_colors = [0, np.random.randint(200, 255), 0, 170]
            meshes.append(leaves)

    print (' done adding trees')
    return meshes

