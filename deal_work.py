import os
import numpy as np
from osgeo import gdal, ogr, osr


creation = ['TILED=YES', 'COMPRESS=DEFLATE',
            'ZLEVEL=3', 'PREDICTOR=1', 'BIGTIFF=YES']
config = ["GDAL_CACHE_MAX=128"]


def rep_file(cache_dir, filename):
    prefix, extension = os.path.splitext(os.path.basename(filename))
    file_path = os.path.join(cache_dir, prefix + extension)
    if os.path.exists(file_path):
        i = 1
        while True:
            file_path = os.path.join(
                cache_dir, prefix + '(' + str(i) + ')' + extension)
            if os.path.exists(file_path):
                i += 1
            else:
                break
    return file_path


def geo2imagexy(ds, x, y):
    trans = ds.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    col, row = np.linalg.solve(a, b) - 0.5
    return int(round(col)), int(round(row))


def extent(ds, layers):
    # clip extent
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if not isinstance(layers, list):
        layers = [layers]
    if isinstance(layers[0], str):
        shp_in = [driver.Open(l) for l in layers]
        layers = [s.GetLayer() for s in shp_in]
    win = np.array([l.GetExtent() for l in layers])
    bound = [win[:, 0:2].min(), win[:, 2:].min(),
             win[:, 0:2].max(), win[:, 2:].max()]
    bound_srs = layers[0].GetSpatialRef()
    return expand_extent(ds, bound, bound_srs), bound_srs


def expand_extent(ds, bound, bound_srs="+proj=longlat +datum=WGS84 +ellps=WGS84"):
    t = ds.GetGeoTransform()
    x_min, y_min, x_max, y_max = proj_bound(ds, bound, bound_srs)
    ulX, ulY = geo2imagexy(ds, x_min, y_min)
    lrX, lrY = geo2imagexy(ds, x_max, y_max)
    clip_range = [min(ulX, lrX), min(ulY, lrY),
                  abs(ulX - lrX) + 1, abs(ulY - lrY) + 1]
    ul_lon = t[0] + t[1] * clip_range[0] + t[2] * clip_range[1]
    ul_lat = t[3] + t[4] * clip_range[0] + t[5] * clip_range[1]
    lr_lon = t[0] + t[1] * (clip_range[0] + clip_range[2]) + \
        t[2] * (clip_range[1] + clip_range[3])
    lr_lat = t[3] + t[4] * (clip_range[0] + clip_range[2]) + \
        t[5] * (clip_range[1] + clip_range[3])
    bound = [min(ul_lon, lr_lon), min(ul_lat, lr_lat),
             max(ul_lon, lr_lon), max(ul_lat, lr_lat)]
    return bound


def proj_bound(ds, bound, bound_srs):
    x_min, y_min, x_max, y_max = bound
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(min(x_min, x_max), min(y_min, y_max))
    ring.AddPoint(max(x_min, x_max), min(y_min, y_max))
    ring.AddPoint(max(x_min, x_max), max(y_min, y_max))
    ring.AddPoint(min(x_min, x_max), max(y_min, y_max))
    ring.AddPoint(min(x_min, x_max), min(y_min, y_max))

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    # output SpatialReference
    if isinstance(bound_srs, osr.SpatialReference):
        outSpatialRef = bound_srs
    else:
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromProj4(bound_srs)

    # raster spatial reference
    ras_srs = osr.SpatialReference()
    ras_srs.ImportFromWkt(ds.GetProjection())

    # create the CoordinateTransformation
    trans = osr.CoordinateTransformation(outSpatialRef, ras_srs)
    trans_reverse = osr.CoordinateTransformation(ras_srs, outSpatialRef)

    # create a geometry from coordinates
    t = ds.GetGeoTransform()
    point = ogr.Geometry(ogr.wkbPoint)
    point_dx = ogr.Geometry(ogr.wkbPoint)
    point_dy = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(t[0] + ds.RasterXSize // 2 * t[1] + ds.RasterYSize // 2 *
                   t[2], t[3] + ds.RasterXSize // 2 * t[4] + ds.RasterYSize // 2 * t[5])
    point_dx.AddPoint(t[0] + (ds.RasterXSize // 2 + 1) * t[1] + ds.RasterYSize // 2 *
                      t[2], t[3] + (ds.RasterXSize // 2 + 1) * t[4] + ds.RasterYSize // 2 * t[5])
    point_dy.AddPoint(t[0] + ds.RasterXSize // 2 * t[1] + (ds.RasterYSize // 2 + 1) *
                      t[2], t[3] + ds.RasterXSize // 2 * t[4] + (ds.RasterYSize // 2 + 1) * t[5])
    point.Transform(trans_reverse)
    point_dx.Transform(trans_reverse)
    point_dy.Transform(trans_reverse)
    dx = abs(point.GetPoint()[0] - point_dx.GetPoint()[0])
    dy = abs(point.GetPoint()[1] - point_dy.GetPoint()[1])

    # density geom
    geom = ogr.CreateGeometryFromWkb(poly.ExportToWkb())
    geom.Segmentize(min(dx, dy) / 2)
    geom.Transform(trans)

    # get boundary
    win = np.array([geom.GetEnvelope()])
    bound = [win[:, 0:2].min(), win[:, 2:].min(),
             win[:, 0:2].max(), win[:, 2:].max()]
    return bound


def convert_uint8(ds, no_data=None):
    if isinstance(ds, str):
        ras = ds
        ds = gdal.Open(ras)
    else:
        ras = ds.GetDescription()

    if ds.ReadAsArray(0, 0, 1, 1).dtype != np.int8:
        return

    if ds.GetRasterBand(1).GetNoDataValue() is not None:
        no_data = ds.GetRasterBand(1).GetNoDataValue()
    if no_data is None:
        raise(ValueError("no_data must be initialed"))

    creation = ['TILED=YES', 'COMPRESS=DEFLATE',
                'ZLEVEL=3', 'PREDICTOR=1', 'BIGTIFF=YES']
    option = gdal.WarpOptions(multithread=True,
                              creationOptions=creation,
                              outputType=gdal.GDT_Byte)
    out_file = rep_file(os.path.dirname(ras), ras)
    ds_out = gdal.Warp(out_file, ras, options=option)

    for i in range(1, 1 + ds.RasterCount):
        band = ds.GetRasterBand(i)
        band_out = ds_out.GetRasterBand(i)
        if band.GetNoDataValue() is not None:
            tile_nodata = band.GetNoDataValue()
        else:
            band.SetNoDataValue(no_data)
            tile_nodata = no_data
        mask = band.ReadAsArray() == tile_nodata
        data = band_out.ReadAsArray()
        data[mask] = 255
        band_out.SetNoDataValue(255)
        band_out.WriteArray(data)

    ds = None
    ds_out = None
    gdal.GetDriverByName('GTiff').Delete(ras)
    os.rename(out_file, ras)


def shp_to_raster(shp, attr, out_path, ds_eg, tem_path, **kwargs):
    # create out put name
    out_file = os.path.join(out_path, os.path.splitext(
        os.path.basename(shp))[0] + '.tif')
    if os.path.exists(out_file):
        return
    tem_file = os.path.join(tem_path, os.path.splitext(
        os.path.basename(shp))[0] + '.tif')

    # extent warp options
    ds_ex = gdal.Translate('/vsimem/_extent.tif', ds_eg, bandList=[1])
    t = ds_eg.GetGeoTransform()
    temp_option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                                   creationOptions=creation, **kwargs,
                                   xRes=t[1] / 10, yRes=t[5] / 10,
                                   outputType=gdal.GDT_Float64)

    ds_tem = gdal.Warp(tem_file, ds_ex, options=temp_option)
    band = ds_tem.GetRasterBand(1)
    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              creationOptions=creation,  **kwargs,
                              xRes=t[1], yRes=t[5], resampleAlg=gdal.GRA_Average,
                              outputType=gdal.GDT_Float64)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_factor = driver.Open(shp)
    layer = shp_factor.GetLayer()

    # create and use RasterizeLayer
    band.Fill(band.GetNoDataValue())
    gdal.RasterizeLayer(ds_tem, [1], layer,
                        options=["ATTRIBUTE=%s" % attr, 'ALL_TOUCHED=TRUE'])
    ds_out = gdal.Warp(out_file, ds_tem, options=option)

    # deal with units
    if os.path.splitext(os.path.basename(out_file))[0] == 'permeability':
        band_out = ds_out.GetRasterBand(1)
        no_data = band_out.GetNoDataValue()
        values = ds_out.ReadAsArray()
        values[values != no_data] = values[values != no_data] / 100
        band_out.WriteArray(values)

        band = ds_tem.GetRasterBand(1)
        no_data = band.GetNoDataValue()
        values = ds_tem.ReadAsArray()
        values[values != no_data] = values[values != no_data] / 100
        band.WriteArray(values)

    band_out = None
    ds_out = None
    band = None
    ds_tem = None
    shp_factor = None
    layer = None


def forest_fraction(ras, out_path, bound, bound_srs,
                    ids, tem_path, **kwargs):
    out_file = os.path.join(out_path, os.path.splitext(
        os.path.basename(ras))[0] + '.tif')
    if os.path.exists(out_file):
        return
    tem_file = os.path.join(tem_path, os.path.splitext(
        os.path.basename(ras))[0] + '.tif')

    ds_in = gdal.Open(ras)
    bound_in = expand_extent(ds_in, bound, bound_srs)
    option = gdal.WarpOptions(multithread=True, creationOptions=creation,
                              outputBounds=bound_in, dstNodata=2)
    ds = gdal.Warp(tem_file, ds_in, options=option)
    band = ds.GetRasterBand(1)
    cover = ds.ReadAsArray()
    is_forest = np.isin(cover, ids).astype(cover.dtype)
    band.WriteArray(is_forest)

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              creationOptions=creation, outputBounds=bound,
                              resampleAlg=gdal.GRA_Average, **kwargs,
                              outputType=gdal.GDT_Float64)
    gdal.Warp(out_file, ds, options=option)


def mean_rasters(ras1, ras2, out_file, bound, bound_srs,
                 no_data, tem_path, **kwargs):
    if os.path.exists(out_file):
        return
    tem_file = os.path.join(tem_path, os.path.splitext(
        os.path.basename(out_file))[0] + '.tif')

    ds_in = gdal.Open(ras1)
    bound_in = expand_extent(ds_in, bound, bound_srs)
    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              creationOptions=creation, outputType=gdal.GDT_Float64,
                              dstNodata=no_data, outputBounds=bound_in)
    ds1 = gdal.Warp(tem_file, ras1, options=option)
    ds2 = gdal.Warp('/vsimem/_2.tif', ras2, options=option)
    count = min(ds1.RasterCount, ds2.RasterCount)
    for c in range(1, 1 + count):
        band1 = ds1.GetRasterBand(c)
        band2 = ds2.GetRasterBand(c)
        arr1 = band1.ReadAsArray()
        arr2 = band2.ReadAsArray()
        mask = (arr1 != no_data) & (arr2 != no_data)
        arr1[mask] = (arr1[mask] + arr2[mask]) / 2
        arr1[~mask] = no_data
        band1.WriteArray(arr1)
    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              creationOptions=creation, outputBounds=bound,
                              resampleAlg=gdal.GRA_Average, **kwargs,
                              outputType=gdal.GDT_Float64)
    gdal.Warp(out_file, ds1, options=option)


def downscaling(ras, out_path, bound, bound_srs, tem_path, **kwargs):
    out_file = os.path.join(out_path, os.path.splitext(
        os.path.basename(ras))[0] + '.tif')
    if os.path.exists(out_file):
        return
    tem_file = os.path.join(tem_path, os.path.splitext(
        os.path.basename(ras))[0] + '.tif')

    ds_in = gdal.Open(ras)
    bound_in = expand_extent(ds_in, bound, bound_srs)
    if os.path.splitext(os.path.basename(ras))[1] == '.grib':
        option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                                  creationOptions=creation,
                                  outputBounds=bound_in,
                                  dstSRS=kwargs['dstSRS'])
    else:
        option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                                  creationOptions=creation,
                                  outputBounds=bound_in)

    ds_tem = gdal.Warp(tem_file, ds_in, options=option)

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              creationOptions=creation, **kwargs,
                              outputBounds=bound,
                              resampleAlg=gdal.GRA_Average)
    # downscaling
    gdal.Warp(out_file, ds_tem, options=option)


def deal_LAI(out_path, **kwargs):
    LAI_file = os.path.join(out_path, 'leaf_area_index.tif')
    if os.path.exists(LAI_file):
        return
    high_cover_ds = gdal.Open('..\\Data\\Raw\\Time\\high_vegetation_cover.grib')
    low_cover_ds = gdal.Open('..\\Data\\Raw\\Time\\low_vegetation_cover.grib')
    high_LAI_ds = gdal.Open(
        '..\\Data\\Raw\\Time\\leaf_area_index_high_vegetation.grib')
    low_LAI_ds = gdal.Open(
        '..\\Data\\Raw\\Time\\leaf_area_index_low_vegetation.grib')

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              **kwargs, creationOptions=creation)
    LAI_ds = gdal.Warp(LAI_file, high_LAI_ds, options=option)
    high_cover_ds = gdal.Warp('/vsimem/_1.tif', high_cover_ds, options=option)
    low_cover_ds = gdal.Warp('/vsimem/_2.tif', low_cover_ds, options=option)
    high_LAI_ds = gdal.Warp('/vsimem/_3.tif', high_LAI_ds, options=option)
    low_LAI_ds = gdal.Warp('/vsimem/_4.tif', low_LAI_ds, options=option)

    # change LAI
    col = LAI_ds.RasterCount
    for c in range(1, col + 1):
        high_cover_band = high_cover_ds.GetRasterBand(c)
        low_cover_band = low_cover_ds.GetRasterBand(c)
        high_LAI_band = high_LAI_ds.GetRasterBand(c)
        low_LAI_band = low_LAI_ds.GetRasterBand(c)
        LAI_band = LAI_ds.GetRasterBand(c)

        high_cover = high_cover_band.ReadAsArray()
        low_cover = low_cover_band.ReadAsArray()
        high_LAI = high_LAI_band.ReadAsArray()
        low_LAI = low_LAI_band.ReadAsArray()

        mask = high_cover != high_cover_band.GetNoDataValue()
        LAI = np.copy(high_cover)
        LAI[mask] = high_cover[mask] * high_LAI[mask] + \
            low_cover[mask] * low_LAI[mask]

        LAI_band.WriteArray(LAI)

    # destroy dataset
    LAI_band = None
    LAI_ds = None


def deal_net(out_path, **kwargs):
    net_file = os.path.join(out_path, 'surface_net_radiation.tif')
    if os.path.exists(net_file):
        return

    solar_ds = gdal.Open('..\\Data\\Raw\\Time\\surface_net_solar_radiation.grib')
    therm_ds = gdal.Open('..\\Data\\Raw\\Time\\surface_net_thermal_radiation.grib')

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              **kwargs, creationOptions=creation)
    net_ds = gdal.Warp(net_file, solar_ds, options=option)
    solar_ds = gdal.Warp('/vsimem/_1.tif', solar_ds, options=option)
    therm_ds = gdal.Warp('/vsimem/_2.tif', therm_ds, options=option)

    # change net radiation
    col = net_ds.RasterCount
    for c in range(1, col + 1):
        solar_band = solar_ds.GetRasterBand(c)
        therm_band = therm_ds.GetRasterBand(c)
        net_band = net_ds.GetRasterBand(c)

        solar = solar_band.ReadAsArray()
        therm = therm_band.ReadAsArray()

        mask = solar != solar_band.GetNoDataValue()
        net = np.copy(solar)
        net[mask] = (solar[mask] + therm[mask]) / 1000000
        net_band.WriteArray(net)

    # destroy dataset
    net_band = None
    net_ds = None


def deal_calc(ras, out_path, plus=None, multi=None, **kwargs):
    out_file = os.path.join(out_path, os.path.splitext(
        os.path.basename(ras))[0] + '.tif')
    if os.path.exists(out_file):
        return

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              creationOptions=creation, **kwargs)
    out_ds = gdal.Warp(out_file, ras, options=option)
    for c in range(1, 1 + out_ds.RasterCount):
        out_band = out_ds.GetRasterBand(c)
        no_data = out_band.GetNoDataValue()
        values = out_band.ReadAsArray()
        if plus is not None:
            values[values != no_data] = values[values != no_data] + plus
        if multi is not None:
            values[values != no_data] = values[values != no_data] * multi
        out_band.WriteArray(values)


def deal_others(ras, out_path, **kwargs):
    out_file = os.path.join(out_path, os.path.splitext(
        os.path.basename(ras))[0] + '.tif')
    if os.path.exists(out_file):
        return

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              creationOptions=creation, **kwargs)
    # downscaling
    gdal.Warp(out_file, ras, options=option)
