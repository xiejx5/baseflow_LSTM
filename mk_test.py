import os
import ogr
import gdal
import numpy as np
import pymannkendall as mk
from clip import clip_with_shp
from collections import Iterable


GRIB_NODATA = -1.797693e+308
creation = ['TILED=YES', 'COMPRESS=DEFLATE',
            'ZLEVEL=3', 'PREDICTOR=1', 'BIGTIFF=YES']


# mann kendall test
def mk_test(arr):
    return mk.original_test(arr).z


def create_tif(filename, ras, values):
    if isinstance(ras, str):
        ds = gdal.Open(ras)
    else:
        ds = ras
        ras = ds.GetDescription()

    if os.path.isdir(os.path.dirname(filename)):
        out_file = filename
    else:
        out_file = os.path.join(os.path.dirname(ras), filename)

    if os.path.exists(out_file):
        return

    count = values.shape[2] if len(values.shape) > 2 else 1
    out_ds = gdal.GetDriverByName('GTiff').Create(
        out_file, ds.RasterXSize, ds.RasterYSize, count, gdal.GDT_Float64, creation)

    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    for c in range(1, 1 + count):
        out_band = out_ds.GetRasterBand(c)
        out_band.SetNoDataValue(GRIB_NODATA)
        if len(values.shape) > 2:
            out_band.WriteArray(values[:, :, c - 1])
        else:
            out_band.WriteArray(values)
    out_band = None
    out_ds = None


def proj_ds(ras, shp, clip_shp=None):
    if isinstance(ras, str):
        ds = gdal.Open(ras)
    else:
        ds = ras
        ras = ds.GetDescription()

    out_file = os.path.join(os.path.dirname(
        ras), os.path.splitext(os.path.basename(ras))[0] + '_proj.tif')
    if os.path.exists(out_file):
        return

    if clip_shp is not None:
        clip_ds = clip_with_shp(ras, clip_shp,
                                out_file='/vsimem/_clip.tif',
                                rasterize_option=['ALL_TOUCHED=False'])
    else:
        clip_ds = ds
    outDataSet = ogr.Open(shp)
    srs = outDataSet.GetLayer().GetSpatialRef()

    option = gdal.WarpOptions(creationOptions=creation,
                              resampleAlg=gdal.GRA_Average,
                              multithread=True, dstSRS=srs)
    gdal.Warp(out_file, clip_ds, options=option)


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
            if isinstance(multi, Iterable):
                values[values != no_data] = \
                    values[values != no_data] * multi[c - 1]
            else:
                values[values != no_data] = values[values != no_data] * multi

        out_band.WriteArray(values)


def deal_divide(filename, ds_up, ds_down, ** kwargs):
    if isinstance(ds_up, str):
        ras_up = ds_up
        ds_up = gdal.Open(ds_up)
    else:
        ras_up = ds_up.GetDescription()

    if isinstance(ds_down, str):
        ds_down = gdal.Open(ds_down)

    if os.path.isdir(os.path.dirname(filename)):
        out_file = filename
    else:
        out_file = os.path.join(os.path.dirname(ras_up), filename)

    if os.path.exists(out_file):
        return

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              **kwargs, creationOptions=creation)
    ds = gdal.Warp(out_file, ds_up, options=option)
    ds_up = gdal.Warp('/vsimem/_1.tif', ds_up, options=option)
    ds_down = gdal.Warp('/vsimem/_2.tif', ds_down, options=option)

    # change net radiation
    col = ds.RasterCount
    for c in range(1, col + 1):
        band_up = ds_up.GetRasterBand(c)
        band_down = ds_down.GetRasterBand(c)
        band = ds.GetRasterBand(c)

        up = band_up.ReadAsArray()
        down = band_down.ReadAsArray()

        mask = up != band_up.GetNoDataValue()
        net = np.copy(up)
        net[mask] = up[mask] / down[mask]
        band.WriteArray(net)

    # destroy dataset
    band = None
    ds = None


def deal_cover(filename, ds_up, ds_down, ** kwargs):
    if isinstance(ds_up, str):
        ras_up = ds_up
        ds_up = gdal.Open(ds_up)
    else:
        ras_up = ds_up.GetDescription()

    if isinstance(ds_down, str):
        ds_down = gdal.Open(ds_down)

    if os.path.isdir(os.path.dirname(filename)):
        out_file = filename
    else:
        out_file = os.path.join(os.path.dirname(ras_up), filename)

    if os.path.exists(out_file):
        return

    option = gdal.WarpOptions(multithread=True, options=["GDAL_CACHE_MAX=128"],
                              **kwargs, creationOptions=creation)
    ds = gdal.Warp(out_file, ds_up, options=option)

    # change net radiation
    col = ds.RasterCount
    for c in range(1, col + 1):
        band_up = ds_up.GetRasterBand(c)
        band_down = ds_down.GetRasterBand(c)
        band = ds.GetRasterBand(c)

        up = band_up.ReadAsArray()
        down = band_down.ReadAsArray()

        mask = up == band_up.GetNoDataValue()
        net = np.copy(down)
        net[mask] = band_up.GetNoDataValue()
        band.SetNoDataValue(band_up.GetNoDataValue())
        band.WriteArray(net)

    # destroy dataset
    band = None
    ds = None
