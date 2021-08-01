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
from pymartini import Martini
from skimage.transform import resize
import math



#address = '7 donnelly rd spencer, ma'
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
    if buffer is None:

        sql = f"""SELECT st_asgeojson(st_intersection(geom, 
                st_makeenvelope({minlon}, {minlat}, {maxlon}, {maxlat}, 4326)))  
          :: json->'coordinates' AS coordinates
          FROM
            publicdata.{table} where 
            pc_centerlon > {minlon} and 
            pc_centerlon < {maxlon} and 
            pc_centerlat > {minlat} and 
            pc_centerlat < {maxlat} limit 3000;"""

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
    coords = [np.array(rec[0][0]) for rec in records if rec is not None and rec[0] is not None]
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
            mesh.visual.mesh.visual.face_colors = [red, red, red, 200]
        else:
            mesh.visual.mesh.visual.face_colors = [*color, 200]
    return meshes

def downloadRasters(centerlon, centerlat, minlon, minlat, maxlon, maxlat):
    sameLocation = False

    try:
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
    vertices, triangles = tile.get_mesh(.0005)
    print ('done')

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
    #triangles = np.flip(triangles, axis=1)

    terrain = trimesh.base.Trimesh(vertices=verts, faces=triangles)

    terrain.visual.mesh.visual.face_colors = [200, 200, 200, 200]

    return [terrain]

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
    meshes.extend(buildLayer(cursor, 'road', minlon, maxlon, minlat, maxlat, elevation, buffer=5, gscale=1.0, color=(0,0,0)))
    print ('adding layer water')
    meshes.extend(buildLayer(cursor, 'water', minlon, maxlon, minlat, maxlat, elevation, gscale=7.5, color=(0,122,255)))
    print ('creating terrain mesh')
    meshes.extend(buildTerrain(elevation, minlon, maxlon, minlat, maxlat, scale = 2.0))

    mesh = trimesh.util.concatenate(meshes)
    #mesh.show()
    #trimesh.exchange.export.export_mesh(mesh, 'test.glb', file_type='glb')


    pyrenderMesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(pyrenderMesh)
    #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1500.0/700.0)
    #pose = transforms3d.affines.compose(np.zeros(3), np.eye(3), np.ones(3), np.zeros(3))

    #scene.add(camera, pose=pose)
    pyrender.Viewer(scene, viewport_size=(1500, 700), use_raymond_lighting=True, shadows=True)


renderScene()

#WITH segments AS (
#SELECT _gid, ST_AsText(ST_MakeLine(lag((pt).geom, 1, NULL) OVER (PARTITION BY _gid ORDER BY _gid, (pt).path), (pt).geom)) AS geom
#  FROM (SELECT _gid, ST_DumpPoints(st_segmentize(geom, 0.0003)) AS pt FROM publicdata.road where pc_centerlon > -71.8186703935037 and
#      pc_centerlon < -71.79867039350368 and
#      pc_centerlat > 42.264305900000004 and
#      pc_centerlat < 42.2843059) as dumps
#)
#SELECT * FROM segments WHERE geom IS NOT NULL;


# WITH segments AS (
# SELECT _gid, ST_AsText(ST_MakeLine(lag((pt).geom, 1, NULL) OVER (PARTITION BY _gid ORDER BY _gid, (pt).path), (pt).geom)) AS geom
#   FROM (SELECT _gid, ST_DumpPoints(st_segmentize(geom, 0.0003)) AS pt FROM publicdata.road where pc_centerlon > -71.8186703935037 and
#       pc_centerlon < -71.79867039350368 and
#       pc_centerlat > 42.264305900000004 and
#       pc_centerlat < 42.2843059) as dumps
# )
# SELECT st_buffer(geom, 0.0001) FROM segments WHERE geom IS NOT NULL;