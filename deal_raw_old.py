import os
import glob
from osgeo import gdal, ogr, osr
from projection import proj_shapefile
from multiprocessing import freeze_support, Pool
from deal_work import (extent, deal_net, deal_LAI, deal_calc, deal_others,
                       shp_to_raster, downscaling, convert_uint8,
                       forest_fraction, mean_rasters)
from _const import CREATION, CONFIG, GRIB_NODATA


# convert int8 to uint8
rasters = glob.glob('..\\Data\\Raw\\Cond\\*.tif') + \
    glob.glob('..\\Data\\Raw\\Cond\\*.grib')

if __name__ == '__main__':
    freeze_support()
    with Pool(4) as p:
        p.map(convert_uint8, rasters)


# bound extent of regions
driver = ogr.GetDriverByName('ESRI Shapefile')
ds_eg = gdal.Open('..\\Data\\Raw\\2m_temperature.grib')
t = ds_eg.GetGeoTransform()
regions = glob.glob('..\\Data\\Ecoregions_wgs84\\*.shp')
bound, bound_srs = extent(ds_eg, regions)

# spatial reference of regions
out_shp = '/vsimem/outline.shp'
outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromWkt(ds_eg.GetProjection())
proj_shapefile(regions[0], out_shp, out_proj=outSpatialRef)
outDataSet = ogr.Open(out_shp)
srs = outDataSet.GetLayer().GetSpatialRef()
outDataSet = None


# shp to conditional raster
out_path = '..\\Data\\Raster_Cond'
if not os.path.exists(out_path):
    os.mkdir(out_path)
tem_path = '..\\Data\\Raw\\Factor'
if not os.path.exists(tem_path):
    os.mkdir(tem_path)
shps = glob.glob('..\\Data\\Raw\\Cond\\*.shp')
attrs = ['logK_Ferr_']
for shp, attr in zip(shps, attrs):
    shp_to_raster(shp, attr, out_path, ds_eg, tem_path,
                  outputBounds=bound, dstSRS=srs, dstNodata=GRIB_NODATA)

# generate fraction forest
ras = '..\\Data\\Raw\\Cond\\forest_fraction.tif'
ids = [40, 50, 60, 70, 90, 100, 110, 160]  # Dong (2012, RSE)
forest_fraction(ras, out_path, bound, bound_srs, ids, tem_path,
                dstSRS=srs, xRes=t[1], yRes=t[5], dstNodata=GRIB_NODATA)

# mean of subsoil and topsoil
soil_type = ['GRAVEL', 'CLAY', 'SAND', 'SILT']
rasters1 = ['..\\Data\\Raw\\Cond\\T_' + i + '.tif' for i in soil_type]
rasters2 = ['..\\Data\\Raw\\Cond\\S_' + i + '.tif' for i in soil_type]
out_files = ['..\\Data\\Raster_Cond\\' + i + '.tif' for i in soil_type]
for ras1, ras2, out_file in zip(rasters1, rasters2, out_files):
    mean_rasters(ras1, ras2, out_file, bound, bound_srs,
                 GRIB_NODATA, tem_path, dstSRS=srs, xRes=t[1], yRes=t[5])

# change resolution of conditional rasters
others = ['dem', 'depth_to_bedrock', 'slope']
rasters = ['..\\Data\\Raw\\Cond\\' + i + '.tif' for i in others] + \
    glob.glob('..\\Data\\Raw\\Cond\\*.grib')
for ras in rasters:
    downscaling(ras, out_path, bound, bound_srs, tem_path,
                dstSRS=srs, xRes=t[1], yRes=t[5])


# time folder and option
out_path = '..\\Data\\Raster_Time'
if not os.path.exists(out_path):
    os.mkdir(out_path)
raw_path = '..\\Data\\Raw'
warp_option = dict(outputBounds=bound, dstSRS=srs, dstNodata=GRIB_NODATA)


# time LAI data
deal_LAI(out_path, **warp_option)


# time radiation data
deal_net(out_path, **warp_option)

# time plus
ras = os.path.join(raw_path, '2m_temperature.grib')
deal_calc(ras, out_path, plus=-273.15, **warp_option)

# time multi
rasters = [os.path.join(raw_path, 'snowfall.grib'),
           os.path.join(raw_path, 'total_precipitation.grib')]
for ras in rasters:
    deal_calc(ras, out_path, multi=1000, **warp_option)


# generate other time data
ras = os.path.join(raw_path, '10m_wind_speed.grib')
deal_others(ras, out_path, **warp_option)
