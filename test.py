import numpy as np
import psycopg2
import geocoder
import trimesh
import random
from gcube import public, globimage, context
import glob
import os
import xarray as xr
import pickle
import pyrender
import transforms3d

address = '100 institute rd worcester, ma'
width, height = 1000, 1000
radius = 0.01
diameter = radius * 2.0
output = "c:/art/out/test.png"
offscreen = False
baseRasterPath = 'c:/art/raster'

def initDBConnection():
    conn = psycopg2.connect(
        host="papercrane-data.ch88iboltgdp.us-east-2.rds.amazonaws.com",
        database="data",
        user="postgres",
        password="EbvanYJnl0F3Jqqgg3Nx")

    cursor = conn.cursor()
    return conn, cursor

def getLocationCoordinates(address):
    g = geocoder.osm(address).json
    centerlon, centerlat = g['lng'], g['lat']

    minlon = centerlon - radius
    minlat = centerlat - radius
    maxlon = centerlon + radius
    maxlat = centerlat + radius

    return centerlon, centerlat, minlon, minlat, maxlon, maxlat

def getTableCoordinates(cursor, table, minlon, maxlon, minlat, maxlat, buffer=None):
    sql = f"SELECT "
    if buffer is None:
        sql+= "ST_AsGeoJSON(geom) :: json->'coordinates' AS coordinates "
    else:
        sql+= f"ST_AsGeoJSON(st_multi(st_buffer(st_segmentize(geom, 0.00027), " \
              f"{buffer / 111000.0}))) :: json->'coordinates' AS coordinates "
    sql+= f"""
    FROM
      publicdata.{table} where 
      pc_centerlon > {minlon} and 
      pc_centerlon < {maxlon} and 
      pc_centerlat > {minlat} and 
      pc_centerlat < {maxlat} limit 3000;"""


    cursor.execute(sql)
    records = cursor.fetchall()
    coords = [np.array(rec[0][0][0]) for rec in records]
    return coords

def buildLayer(cursor, table, minlon, maxlon, minlat, maxlat, elevation, buffer=None, gscale=1.0,
               color=None):

    meshes = []
    coordsList = getTableCoordinates(cursor, table, minlon, maxlon, minlat, maxlat, buffer)
    print ('  retrieved table coordinates')
    centerCoord = np.array([(minlon + maxlon) / 2.0, (minlat + maxlat) / 2.0])
    xx = []
    yy = []
    for coordsl in coordsList:

        lon, lat = np.mean(coordsl, axis=0)
        xx.append(lon)
        yy.append(lat)
        coords = (coordsl[:-1] - centerCoord) * (111000.0)
        numCoords = coords.shape[0]
        edges = np.array([np.arange(0, numCoords), np.arange(1, numCoords+1)]).T
        edges[-1,1] = 0

        poly = trimesh.path.polygons.edges_to_polygons(edges, coords)[0]
        scale = trimesh.path.polygons.polygon_scale(poly)
        mesh = trimesh.creation.extrude_polygon(poly, scale*gscale)
        meshes.append(mesh)

    xx = xr.DataArray(xx, dims="points")
    yy = xr.DataArray(yy, dims="points")
    elevs = elevation.interp(x=xx,y=yy, method='nearest').values
    print (elevs.shape)
    print (' done sampling the elevation')
    print ('translating meshes')
    for mesh, elev in zip(meshes, elevs):
        mesh.vertices[:, 2]+=elev * 2.0
        if color is None:
            red = random.randint(0, 255)
            mesh.visual.mesh.visual.face_colors = [red, red, red, 200]
        else:
            mesh.visual.mesh.visual.face_colors = [*color, 200]
    return meshes

def downloadRasters(centerlon, centerlat, minlon, minlat, maxlon, maxlat):
    sameLocation = False

    try:
        file = open(os.path.join(baseRasterPath, 'coords.p'), "rb")
        caCenterlon, caCenterlat = pickle.load(open(os.path.join(baseRasterPath, 'coords.p'), "rb"))
        if abs(caCenterlon - centerlon) < 0.0001 and abs(caCenterlat - centerlat) < 0.0001:
            sameLocation = True
    except:
        print('No coords.p file found.')

    if not sameLocation:
        files = glob.glob(os.path.join(baseRasterPath, '*.tif'))
        files.extend(glob.glob(os.path.join(baseRasterPath, '*.xml')))
        files.extend(glob.glob(os.path.join(baseRasterPath, '*.p')))

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

        pickle.dump([centerlon, centerlat], open(os.path.join(baseRasterPath, 'coords.p'), "wb"))
        bbox = [minlon, maxlon, minlat, maxlat]
        ctx = context.Context(args=[], currentProcess='art')
        rasterList, rasterOutList = public.importPublic(ctx, '', bbox, [], 0, {'elevation': 'elevation'}, False)
        globimage.importRasters(rasterList, baseRasterPath, bbox, outNames=rasterOutList)


def renderScene():

    meshes = []
    conn, cursor = initDBConnection()
    centerlon, centerlat, minlon, minlat, maxlon, maxlat = getLocationCoordinates(address)

    downloadRasters(centerlon, centerlat, minlon, minlat, maxlon, maxlat)
    elevation = xr.open_rasterio(os.path.join(baseRasterPath, 'elevation.tif'))[0, :, :]

    meshes = []
    print ('adding layer building')
    meshes.extend(buildLayer(cursor, 'building', minlon, maxlon, minlat, maxlat, elevation))
    print ('adding layer road')
    meshes.extend(buildLayer(cursor, 'road', minlon, maxlon, minlat, maxlat, elevation, buffer=5, gscale=0.002, color=(0,0,0)))
    print ('adding layer water')
    meshes.extend(buildLayer(cursor, 'water', minlon, maxlon, minlat, maxlat, elevation, gscale=0.002, color=(0,0,255)))


    mesh = trimesh.util.concatenate(meshes)
    trimesh.exchange.export.export_mesh(mesh, 'test.glb', file_type='glb')


    pyrenderMesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(pyrenderMesh)
    #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1500.0/700.0)
    #pose = transforms3d.affines.compose(np.zeros(3), np.eye(3), np.ones(3), np.zeros(3))

    #scene.add(camera, pose=pose)
    pyrender.Viewer(scene, viewport_size=(1500, 700), use_raymond_lighting=True, shadows=True)


renderScene()

