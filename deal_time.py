import os
import glob
import shutil
import geotools as gt
from functools import partial
import numpy as np
from osgeo import gdal
from multiprocessing import freeze_support, Pool, cpu_count
from _const import (GRIB_NODATA, RAW_COND, RAW_TIME, RAS_TIME, ECO_WGS)


# example time series raster
ds_eg = gdal.Open(os.path.join(RAW_TIME, '2m_temperature.grib'))
t = ds_eg.GetGeoTransform()


# get study bound and grib file spatial reference
regions = glob.glob(os.path.join(ECO_WGS, '*.shp'))
bound, bound_srs = gt.bound_layers(regions)
bound, bound_srs = gt.bound_raster(ds_eg, bound, bound_srs)
bound_srs = "+proj=longlat +datum=WGS84 +ellps=WGS84"


# convert int8 to uint8
rasters = glob.glob(os.path.join(RAW_TIME, '*.tif')) + \
    glob.glob(os.path.join(RAW_TIME, '*.grib')) + \
    glob.glob(os.path.join(RAW_COND, '*.tif')) + \
    glob.glob(os.path.join(RAW_COND, '*.grib'))
if __name__ == '__main__':
    freeze_support()
    with Pool(cpu_count() - 1) as p:
        p.map(gt.convert_uint8, rasters)


# convert grib to tif with wgs 84
grib_rasters = [r for r in rasters if '.grib' in r]
warp_option = dict(outputBounds=bound, dstSRS=bound_srs,
                   dstNodata=GRIB_NODATA, xRes=t[1], yRes=t[5])
if __name__ == '__main__':
    freeze_support()
    with Pool(min(cpu_count() - 1, len(grib_rasters))) as p:
        ds = p.map(partial(gt.grib_to_tif, **warp_option), grib_rasters)


# deal the 8 time series rasters
out_path = RAS_TIME
if not os.path.isdir(out_path):
    os.makedirs(out_path)


# precipitation
ras = os.path.join(RAW_TIME, 'total_precipitation.tif')
ds = gdal.Open(ras)
gt.map_calc(ras, 'A*1000', out_path)


# snowfall
ras = os.path.join(RAW_TIME, 'snowfall.tif')
gt.map_calc(ras, 'A*1000', out_path)


# 2m temperature
ras = os.path.join(RAW_TIME, '2m_temperature.tif')
gt.map_calc(ras, 'A-273.15', out_path)
out_file = gt.context_file(ras, out_path)
if not os.path.exists(out_file):
    gt.Calc('A-273.15', out_file, creation_options=gt.CREATION,
            allBands='A', quiet=True, A=ras)


# 10m wind speed
ras = os.path.join(RAW_TIME, '10m_wind_speed.tif')
if not os.path.exists(out_file):
    shutil.copy2(ras, out_file)


# surface net radiation
rasters = ['surface_net_solar_radiation.tif',
           'surface_net_thermal_radiation.tif']
rasters = [os.path.join(RAW_TIME, r) for r in rasters]
out_file = os.path.join(RAS_TIME, 'surface_net_radiation.tif')
gt.map_calc(rasters, '(A+B)/86400', out_file)


# leaf area index
rasters = ['leaf_area_index_high_vegetation.tif', 'high_vegetation_cover.tif',
           'leaf_area_index_low_vegetation.tif', 'low_vegetation_cover.tif']
rasters = [os.path.join(RAW_TIME, r) for r in rasters]
out_file = os.path.join(RAS_TIME, 'leaf_area_index.tif')
n_band = gdal.Open(rasters[0]).RasterCount
iter_idxs = np.repeat(np.arange(1, 1 + n_band).reshape(-1, 1), 2, axis=1)
stat_idxs = np.ones(iter_idxs.shape)
band_idxs = np.concatenate([iter_idxs, stat_idxs], axis=1)[:, [0, 2, 1, 3]]
gt.map_calc(rasters, 'A*B+C*D', out_file, band_idxs=band_idxs)
