import os
import glob
import pandas as pd
import numpy as np
import geotools as gt
from osgeo import ogr
from _const import GRIB_NODATA


def time_series(rasters, shp, out_path):
    basin_id = os.path.splitext(os.path.basename(shp))[0]
    file_path = os.path.join(out_path, basin_id + '.csv')
    if os.path.exists(file_path):
        return

    df = pd.DataFrame()
    for i, ras in enumerate(rasters):
        ras_name = os.path.splitext(os.path.basename(ras))[0]
        series = np.squeeze(gt.extract(ras, shp, enlarge=10, ext=basin_id, stat=True,
                                       no_data=GRIB_NODATA, save_cache=True, new=False))
        df = pd.concat([df, pd.Series(data=series, name=ras_name)], axis=1)

    df.to_csv(file_path, index=False)


def weighted_mean(in_shp, clip_shp, field, out_shp=None, save_cache=False):
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # get layer of in_shp
    if isinstance(in_shp, str):
        ds = driver.Open(in_shp)
    else:
        ds = in_shp

    in_layer = ds.GetLayer()
    srs = ds.GetLayer().GetSpatialRef()

    # project clip_shp
    if save_cache:
        proj_shp = rep_file('cache', os.path.splitext(
            os.path.basename(clip_shp))[0] + '_proj.shp')
    else:
        proj_shp = '/vsimem/_proj.shp'
    gt.proj_shapefile(clip_shp, proj_shp, out_proj=srs)
    clip_ds = driver.Open(proj_shp)
    clip_layer = clip_ds.GetLayer()

    # export out_shp
    if out_shp is None:
        if save_cache:
            out_shp = rep_file('cache', os.path.splitext(
                os.path.basename(clip_shp))[0] + '_out.shp')
        else:
            out_shp = '/vsimem/_out.shp'

    out_ds = driver.CreateDataSource(out_shp)
    out_layer = out_ds.CreateLayer(out_shp, srs=srs)

    in_layer.Clip(clip_layer, out_layer)

    area = []
    logK = []
    # newField = ogr.FieldDefn('Area', ogr.OFTReal)
    # out_layer.CreateField(newField)
    c = out_layer.GetFeatureCount()
    for i in range(c):
        f = out_layer.GetFeature(i)
        area.append(f.GetGeometryRef().GetArea())
        logK.append(f.GetField(field))
        # f.SetField('Area', f.GetGeometryRef().GetArea())
        # out_layer.SetFeature(f)
    area = np.array(area)
    logK = np.array(logK)
    mean_logK = np.average(logK, weights=area)
    # mean_logK = np.log10(np.average(np.power(10, logK / 100), weights=area))
    out_layer = None
    out_ds = None
    return mean_logK


def cond_factor_orgin(shp):
    basin_id = os.path.splitext(os.path.basename(shp))[0]
    shapes = glob.glob('Raw\\Cond\\*.shp')
    rasters = glob.glob('Raw\\Cond\\*.tif') + \
        glob.glob('Raw\\Cond\\*.grib')
    one_out = np.full(len(shapes) + len(rasters), np.nan)

    attrs = ['logK_Ferr_']
    for i, (in_shp, attrs) in enumerate(zip(shapes, attrs)):
        one_out[i] = weighted_mean(in_shp, shp, attrs)
    # logK / 100
    one_out[0] = one_out[0] / 100

    for i, ras in enumerate(rasters):
        one_out[i + len(shapes)] = np.squeeze(
            gt.extract(ras, shp, stat=True, enlarge=10,
                       ext=basin_id, no_data=GRIB_NODATA))

    return one_out


def cond_factor(rasters, shp):
    basin_id = os.path.splitext(os.path.basename(shp))[0]
    one_out = np.full(len(rasters), np.nan)

    for i, ras in enumerate(rasters):
        one_out[i] = np.squeeze(
            gt.extract(ras, shp, stat=True, enlarge=10,
                       ext=basin_id, no_data=GRIB_NODATA))

    return one_out


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
