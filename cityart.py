import numpy as np
import psycopg2
import trimesh
import random
import glob
import os
import xarray as xr
import pickle
import pyrender
from pymartini import Martini
from skimage.transform import resize
import math
import tree
import raster
import ctypes
import itertools
import os
import string
import platform
import usaddress


#address = '7 donnelly rd: spencer: ma'
#address = '1350 Massachusetts Ave Cambridge MA'
#address = 'chicago, IL'
#address = '266 Harding St Worcester, MA 01610'
address = '166 harding St Worcester MA'
#address = '25 Quincy St: Cambridge: MA'


width, height = 1500, 1000
radius = 0.01
diameter = radius * 2.0
baseRasterPath = 'c:/art/raster'

def initDBConnection():
    conn = psycopg2.connect(
        host="localhost",
        database="data",
        user="postgres",
        password="gcube")

    cursor = conn.cursor()
    return conn, cursor

def getAvailableDrives():
    if 'Windows' not in platform.system():
        return []
    drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
    return list(itertools.compress(string.ascii_uppercase,
               map(lambda x:ord(x) - ord('0'), bin(drive_bitmask)[:1:-1])))


def getLocationCoordinates(address, cursor):

    # convert the address string into formated street, city, state strings.
    results = usaddress.tag(address)[0]
    print (results)
    # OrderedDict([('AddressNumber', '1350'), ('StreetName', 'Massachusetts'), ('StreetNamePostType', 'Ave'), ('PlaceName', 'Cambridge'), ('StateName', 'MA')]
    street = results['AddressNumber'] + ' ' + results['StreetName'] + ' ' + results['StreetNamePostType']
    city = results['PlaceName']
    state = results['StateName']
    street = street.strip().title()
    city = city.strip().title()
    state = state.strip().upper()

    print (street, city, state)
    sql = f"select pc_centerlon, pc_centerlat from publicdata.address where address = '{street}' and city = '{city}'" \
          f" and state = '{state}'"

    print (f'looking up address {street}, {city}, {state}')

    cursor.execute(sql)
    res = cursor.fetchall()

    print ('done')

    centerlon, centerlat = res[0]


    print (centerlon, centerlat)
    minlon = centerlon - radius
    minlat = centerlat - radius
    maxlon = centerlon + radius
    maxlat = centerlat + radius

    return centerlon, centerlat, minlon, minlat, maxlon, maxlat

def getTableCoordinates(cursor, table, minlon, maxlon, minlat, maxlat, buffer=None):
    if buffer is None:

        sql = f"""SELECT st_asgeojson(st_intersection(geom, 
                st_makeenvelope({minlon}, {minlat}, {maxlon}, {maxlat}, 4326)))  
          :: json->'coordinates' AS coordinates
          FROM
            publicdata.{table} where 
            pc_centerlon > {minlon} and 
            pc_centerlon < {maxlon} and 
            pc_centerlat > {minlat} and 
            pc_centerlat < {maxlat};"""

    else:
        sql = f"""WITH segments AS (
        SELECT _gid, ST_AsText(ST_MakeLine(lag((pt).geom, 1, NULL) OVER (PARTITION BY _gid ORDER BY _gid, (pt).path), (pt).geom)) AS geom
        FROM (SELECT _gid, ST_DumpPoints(st_segmentize(geom, 0.00155)) AS pt FROM publicdata.road where pc_centerlon > {minlon} and
           pc_centerlon < {maxlon} and
           pc_centerlat > {minlat} and
           pc_centerlat < {maxlat}) as dumps
        )
        SELECT st_asgeojson(st_intersection(st_setsrid(st_buffer(geom, 0.00007, 2), 4326), 
        st_makeenvelope({minlon}, {minlat}, {maxlon}, {maxlat}, 4326))
        ) :: json->'coordinates' as coordinates  FROM segments WHERE geom IS NOT NULL;
        """

    cursor.execute(sql)
    records = cursor.fetchall()
    coords = [np.array(rec[0][0]) for rec in records if rec is not None and len(rec) > 0 and
              rec[0] is not None and len(rec[0]) > 0]
    return coords

def buildLayer(cursor, table, minlon, maxlon, minlat, maxlat, elevation, buffer=None, gscale=0.0,
               color=None):

    meshes = []
    coordsList = getTableCoordinates(cursor, table, minlon, maxlon, minlat, maxlat, buffer)
    print ('  retrieved table coordinates')
    centerCoord = np.array([(minlon + maxlon) / 2.0, (minlat + maxlat) / 2.0])
    xx = []
    yy = []
    for coordsl in coordsList:

        scoords = coordsl.squeeze()
        lon, lat = np.mean(scoords, axis=0)
        xx.append(lon)
        yy.append(lat)
        coords = (scoords[:-1] - centerCoord) * (111000.0)
        numCoords = coords.shape[0]
        edges = np.array([np.arange(0, numCoords), np.arange(1, numCoords+1)]).T
        edges[-1,1] = 0

        poly = trimesh.path.polygons.edges_to_polygons(edges, coords)[0]
        scale = trimesh.path.polygons.polygon_scale(poly)
        if gscale < 0.001:
            mesh = trimesh.creation.extrude_polygon(poly, scale)
        else:
            mesh = trimesh.creation.extrude_polygon(poly, gscale)
        meshes.append(mesh)

    xx = xr.DataArray(xx, dims="points")
    yy = xr.DataArray(yy, dims="points")

    if len(xx) == 0:
        return []
    elevs = elevation.interp(x=xx,y=yy, method='nearest').values


    print (elevs.shape)
    print (' done sampling the elevation')
    print ('translating meshes')

    for mesh, elev in zip(meshes, elevs):
        melev = elev
        if math.isnan(elev):
            melev = 150.0
        mesh.vertices[:, 2]+=melev * 2.0
        if color is None:
            red = random.randint(0, 255)
            mesh.visual.mesh.visual.face_colors = [red, red, red, 255]
        else:
            mesh.visual.mesh.visual.face_colors = [*color, 255]
    return meshes

def downloadRasters(centerlon, centerlat, minlon, minlat, maxlon, maxlat, cursor):
    sameLocation = False

    rasters = ['elevation', 'tree']


    try:
        caCenterlon, caCenterlat = pickle.load(open(os.path.join(baseRasterPath, 'coords.p'), "rb"))
        if abs(caCenterlon - centerlon) < 0.0005 and abs(caCenterlat - centerlat) < 0.0005:
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

        rasterList = raster.getRasterList(rasters, bbox, cursor)
        drives = getAvailableDrives()
        if 'E' in drives:
            localRasterList = raster.s3ToLocal(rasterList)
        else:
            # The external hard drive is not plugged in so we grab the elevation data directly from S3.
            localRasterList = rasterList
        raster.importRasters(localRasterList, baseRasterPath, bbox, outNames=rasters)

def buildTerrain(elevation, minlon, maxlon, minlat, maxlat, scale = 1.0):

    # we now need to make this array square.
    elx, ely = elevation.shape
    elx = elx-1
    ely = ely-1
    elevationSquare = resize(elevation.values, (1025, 1025)).astype(np.float32).T * scale

    martini = Martini(1025)
    # generate RTIN hierarchy from terrain data (an array of size^2 length)
    tile = martini.create_tile(elevationSquare)

    # get a mesh (vertices and triangles indices) for a 10m error
    print ('generating terrain')
    vertices, triangles = tile.get_mesh(.00003)

    vertices = vertices.reshape((int(vertices.shape[0]/2), 2))
    triangles = triangles.reshape((int(triangles.shape[0]/3), 3))
    elevationSamples = elevationSquare[vertices[:,0], vertices[:,1]]

    verts = np.vstack((vertices.T, elevationSamples)).T

    print (minlon, maxlon, minlat, maxlat)
    mincoords = elevation[0,0].coords
    maxcoords = elevation[elx, ely].coords
    rminx, rminy = mincoords['x'].values, mincoords['y'].values
    rmaxx, rmaxy = maxcoords['x'].values, maxcoords['y'].values

    verts[:, 0] = ((verts[:, 0]/1025.0) - 0.5) * (rmaxx - rminx) * 111000.0
    verts[:, 1] = ((verts[:, 1]/1025.0) - 0.5) * (rmaxy - rminy) * 111000.0

    terrain = trimesh.base.Trimesh(vertices=verts, faces=triangles)

    terrain.visual.mesh.visual.face_colors = [230, 230, 230, 80]

    return [terrain]

def renderScene():


    conn, cursor = initDBConnection()
    centerlon, centerlat, minlon, minlat, maxlon, maxlat = getLocationCoordinates(address, cursor)

    downloadRasters(centerlon, centerlat, minlon, minlat, maxlon, maxlat, cursor)
    elevation = xr.open_rasterio(os.path.join(baseRasterPath, 'elevation.tif'))[0, :, :]

    meshes = []
    print ('adding layer building')
    meshes.extend(buildLayer(cursor, 'building', minlon, maxlon, minlat, maxlat, elevation))
    print ('adding layer road')
    meshes.extend(buildLayer(cursor, 'road', minlon, maxlon, minlat, maxlat,
                             elevation, buffer=5, gscale=2.0, color=(0,0,0)))
    print ('adding layer water')
    meshes.extend(buildLayer(cursor, 'water', minlon, maxlon, minlat, maxlat,
                             elevation, gscale=7.5, color=(0,70,255)))
    print ('creating terrain mesh')
    meshes.extend(buildTerrain(elevation, minlon, maxlon, minlat, maxlat, scale = 2.0))

    meshes.extend(tree.addTrees(elevation, centerlon, centerlat))

    mesh = trimesh.util.concatenate(meshes)
    #mesh.show()
    #trimesh.exchange.export.export_mesh(mesh, 'test.glb', file_type='glb')


    pyrenderMesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(pyrenderMesh)
    #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1500.0/700.0)
    #pose = transforms3d.affines.compose(np.zeros(3), np.eye(3), np.ones(3), np.zeros(3))

    #scene.add(camera, pose=pose)
    pyrender.Viewer(scene, viewport_size=(width, height),
                    use_raymond_lighting=True,
                    shadows=True,
                    window_title = 'World builder v0.1a - David Berthiaume')

    conn.close()

renderScene()

