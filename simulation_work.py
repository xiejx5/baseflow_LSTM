import os
import glob
import numpy as np
from osgeo import gdal, osr
from clip import clip_with_shp
from math import atan, atan2, cos, pi, sin, sqrt, tan, radians
from deal_raw import GRIB_NODATA


# raster data
rasters = glob.glob('..\\Data\\Local\\Time\\*.tif')
raster_names = [os.path.splitext(os.path.basename(f))[0] for f in rasters]
cond_rasters = glob.glob('..\\Data\\Local\\Cond\\*.tif')
cond_names = [os.path.splitext(os.path.basename(f))[0] for f in cond_rasters]


def partition_work(shp):
    eco = os.path.splitext(os.path.basename(shp))[0].replace('_wgs84', '')
    for ras, ras_name in zip(rasters, raster_names):
        out_file = os.path.join(
            '..\\Data\\Raster_Ecoregions', eco + '_' + ras_name + '.tif')
        if not os.path.exists(out_file):
            clip_with_shp(ras, shp, ext=eco, rect_file=out_file,
                          no_data=GRIB_NODATA, save_cache=True, new=False)
    ds_mask = gdal.Open(out_file)
    band = ds_mask.GetRasterBand(1)
    mask = band.ReadAsArray() != band.GetNoDataValue()
    ds_mask = None
    band = None

    for ras, ras_name in zip(cond_rasters, cond_names):
        out_file = os.path.join(
            '..\\Data\\Raster_Ecoregions', eco + '_' + ras_name + '.tif')
        if os.path.exists(out_file):
            continue
        clip_with_shp(ras, shp, ext=eco, rect_file=out_file,
                      no_data=GRIB_NODATA, save_cache=True, new=False)
        fill_mask(out_file, mask)
    return eco


def fill_mask(out_file, mask):
    ds = gdal.Open(out_file, gdal.GA_Update)
    rows, cols = np.where(mask)

    count = ds.RasterCount
    for i in range(1, 1 + count):
        band = ds.GetRasterBand(i)
        no_data = band.GetNoDataValue()
        cond = band.ReadAsArray()
        no_index = np.where(cond[mask] == no_data)[0]
        valid_rows = np.delete(rows, no_index)
        valid_cols = np.delete(cols, no_index)

        # write non mask to no data
        all_no = np.all(cond[~mask] == no_data)
        if not all_no:
            cond[~mask] = no_data

        if not no_index.shape[0] and all_no:
            continue
        if not no_index.shape[0] and not all_no:
            band.WriteArray(cond)
            continue

        for r, c in zip(rows[no_index], cols[no_index]):
            col_width = distance(imagexy2lonlat(ds, r, c),
                                 imagexy2lonlat(ds, r, c + 1))
            row_height = distance(imagexy2lonlat(ds, r, c),
                                  imagexy2lonlat(ds, r + 1, c))
            points_distance = np.sqrt(np.sum(np.square(
                ((valid_rows - r) * row_height,
                 (valid_cols - c) * col_width)), axis=0))
            min_index = np.argmin(points_distance)
            cond[r, c] = cond[valid_rows[min_index],
                              valid_cols[min_index]]
        band.WriteArray(cond)


def getSRSPair(dataset):
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2lonlat(dataset, x, y):
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]


def imagexy2geo(dataset, row, col):
    trans = dataset.GetGeoTransform()
    px = trans[0] + (col + 0.5) * trans[1] + (row + 0.5) * trans[2]
    py = trans[3] + (col + 0.5) * trans[4] + (row + 0.5) * trans[5]
    return px, py


def imagexy2lonlat(ds, row, col):
    geo_xy = imagexy2geo(ds, row, col)
    lon, lat = geo2lonlat(ds, geo_xy[0], geo_xy[1])
    return lon, lat


def distance(a, b):
    ELLIPSOIDS = {
        # model           major (km)   minor (km)     flattening
        'WGS-84':        (6378.137, 6356.7523142, 1 / 298.257223563),
    }

    lng1, lat1 = radians(a[0]), radians(a[1])
    lng2, lat2 = radians(b[0]), radians(b[1])

    major, minor, f = ELLIPSOIDS['WGS-84']

    delta_lng = lng2 - lng1

    reduced_lat1 = atan((1 - f) * tan(lat1))
    reduced_lat2 = atan((1 - f) * tan(lat2))

    sin_reduced1, cos_reduced1 = sin(reduced_lat1), cos(reduced_lat1)
    sin_reduced2, cos_reduced2 = sin(reduced_lat2), cos(reduced_lat2)

    lambda_lng = delta_lng
    lambda_prime = 2 * pi

    iter_limit = 20

    i = 0

    while (i == 0 or
           (abs(lambda_lng - lambda_prime) > 10e-12 and i <= iter_limit)):
        i += 1

        sin_lambda_lng, cos_lambda_lng = sin(lambda_lng), cos(lambda_lng)

        sin_sigma = sqrt(
            (cos_reduced2 * sin_lambda_lng) ** 2 +
            (cos_reduced1 * sin_reduced2 -
             sin_reduced1 * cos_reduced2 * cos_lambda_lng) ** 2
        )

        if sin_sigma == 0:
            return 0  # Coincident points

        cos_sigma = (
            sin_reduced1 * sin_reduced2 +
            cos_reduced1 * cos_reduced2 * cos_lambda_lng
        )

        sigma = atan2(sin_sigma, cos_sigma)

        sin_alpha = (
            cos_reduced1 * cos_reduced2 * sin_lambda_lng / sin_sigma
        )
        cos_sq_alpha = 1 - sin_alpha ** 2

        if cos_sq_alpha != 0:
            cos2_sigma_m = cos_sigma - 2 * (
                sin_reduced1 * sin_reduced2 / cos_sq_alpha
            )
        else:
            cos2_sigma_m = 0.0  # Equatorial line

        C = f / 16. * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))

        lambda_prime = lambda_lng
        lambda_lng = (
            delta_lng + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (
                    cos2_sigma_m + C * cos_sigma * (
                        -1 + 2 * cos2_sigma_m ** 2
                    )
                )
            )
        )

    if i > iter_limit:
        raise ValueError("Vincenty formula failed to converge!")

    u_sq = cos_sq_alpha * (major ** 2 - minor ** 2) / minor ** 2

    A = 1 + u_sq / 16384. * (
        4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq))
    )

    B = u_sq / 1024. * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    delta_sigma = (
        B * sin_sigma * (
            cos2_sigma_m + B / 4. * (
                cos_sigma * (
                    -1 + 2 * cos2_sigma_m ** 2
                ) - B / 6. * cos2_sigma_m * (
                    -3 + 4 * sin_sigma ** 2
                ) * (
                    -3 + 4 * cos2_sigma_m ** 2
                )
            )
        )
    )

    s = minor * A * (sigma - delta_sigma)
    return s
