import os
import numpy as np
from osgeo import gdal, ogr, osr


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
    high_cover_ds = gdal.Open(
        '..\\Data\\Raw\\Time\\high_vegetation_cover.grib')
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

    solar_ds = gdal.Open(
        '..\\Data\\Raw\\Time\\surface_net_solar_radiation.grib')
    therm_ds = gdal.Open(
        '..\\Data\\Raw\\Time\\surface_net_thermal_radiation.grib')

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
