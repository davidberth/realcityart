import os

def getRasterList(rasters, bbox, cursor):

    xmin, xmax, ymin, ymax = bbox

    # Now obtain a list of all tables that intersect our area of interest
    sql = f"""
        select a.public_table, a.geom_type, b.source_filename, raster_cell_size, raster_units  
        from metadata.pubdatainfo a, metadata.rawdatainfo b 
            where st_intersects(a.convex_hull::geometry, 
            st_makeenvelope({xmin}, {ymin}, {xmax}, {ymax}, 4326)) 
         and a.source_table = b.table_name"""
    cursor.execute(sql)
    res = cursor.fetchall()

    rasterList = []

    for i in res:
        tname = i[0]
        gtype = i[1]
        sfile = i[2]

        if tname in rasters:

            if (gtype == 'RASTER' or gtype == 'raster') :
                if sfile == 'various':
                    # we are looking at a tiled raster dataset
                    # we need to see which tiles intersect our bounding box
                    sql = f"""select raster from publicdata.{tname} where 
                        st_intersects(convex_hull, st_makeenvelope({xmin}, {ymin}, {xmax}, {ymax}, 4326))"""

                    cursor.execute(sql)
                    res = cursor.fetchall()

                    rasterOutLocalList = []

                    for qii in res:
                        qii1 = qii[0]
                        rasterNameList = qii1.split('/buckets/')
                        if len(rasterNameList) > 1:
                            rasterOutLocalList.append('/vsis3/' + rasterNameList[1])

                    rasterList.append(sorted(rasterOutLocalList))

                else:
                    rasterNameList = sfile.split('/buckets/')
                    if len(rasterNameList) > 1:
                        rasterName = '/vsis3/' + rasterNameList[1]
                        rasterList.append(rasterName)

    return rasterList

def s3ToLocal(rasterList):

    sourceString = '/vsis3/papercrane-public/'
    targetString = 'e:/imagery/'

    localRasterList = []
    for item in rasterList:
        if isinstance(item, str):
            localRasterList.append( item.replace(sourceString, targetString))
        else:
            localList = []
            for subItem in item:
                localList.append( subItem.replace(sourceString, targetString))
            localRasterList.append(localList)

    return localRasterList

def importRasters(rlist, tempdir, bbox, outNames = None):

    bboxstring = str(bbox[0] - 0.001) + ' ' + str(bbox[3] + 0.001) + ' ' + str(bbox[1] + 0.001) + ' ' + str(bbox[2] - 0.001)
    bboxstring2 = str(bbox[0] - 0.001) + ' ' + str(bbox[2] - 0.001) + ' ' + str(bbox[1] + 0.001) + ' ' + str(bbox[3] + 0.001)

    for e, i in enumerate(rlist):

        if isinstance(i, list):
            bn = outNames[e] + '.tif'
            pbn = os.path.join(tempdir, bn)

            cline = 'gdalwarp  --config GDAL_CACHEMAX 3000 -r near -te_srs EPSG:4326 -te ' + bboxstring2 + ' ' +  i[0] + ' ' + pbn
            os.system(cline)

            for jjj in i[1:]:
                cline = 'gdalwarp --config GDAL_CACHEMAX 3000 -r near ' + jjj + ' ' + pbn
                os.system(cline)

        else:
            bn = os.path.basename(i)
            if outNames is None:
                bn = bn[:-4] + '.tif'
            else:
                bn = outNames[e] + '.tif'
            pbn = os.path.join(tempdir, bn)
            cline = 'gdal_translate -projwin_srs EPSG:4326 -projwin ' + bboxstring + ' ' + i + ' ' + pbn

            try:
                os.system(cline)
            except:
                print(f"skipping {i}")