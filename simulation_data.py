import os
import glob
import numpy as np
import pandas as pd
from osgeo import ogr, gdal
from cond_lstm import TIME_STEPS, INPUT_DIM
from deal_raw import GRIB_NODATA
from clip import clip_with_shp

SIMU_BATCH = 10000

creation = ['TILED=YES', 'COMPRESS=DEFLATE',
            'ZLEVEL=3', 'PREDICTOR=1', 'BIGTIFF=YES']


def treat_ras(ras):
    if isinstance(ras, str):
        ds = gdal.Open(ras)
    else:
        ds = ras
        ras = ds.GetDescription()
    return ds, ras


def get_info_simu(eco):
    var = pd.read_csv(
        next(glob.iglob('..\\Data\\Basin_Time\\*.csv')), nrows=0).columns
    ds = gdal.Open('..\\Data\\Raster_Ecoregions\\' +
                   eco + '_' + var[0] + '.tif')
    band = ds.GetRasterBand(1)
    cells = pd.DataFrame()
    cells['row'], cells['col'] = np.where(
        band.ReadAsArray() != band.GetNoDataValue())
    start_month = ((TIME_STEPS - 2) // 12 + 1) * 12
    end_month = ds.RasterCount // 12 * 12
    cells['num_months'] = end_month - start_month
    cells['t'] = np.cumsum(cells['num_months'])
    cells['s'] = cells['t'].shift(1, fill_value=0)

    return cells, start_month, end_month


def load_data_simu(eco, cells, start_month, end_month, norm):
    var = pd.read_csv(
        next(glob.iglob('..\\Data\\Basin_Time\\*.csv')), nrows=0).columns
    cond_mean, cond_std, X_mean, X_std = norm[0:4]

    # condition
    cond_names = pd.read_excel(
        '..\\Data\\Factor_Cond.xlsx', nrows=0).columns
    factor = pd.DataFrame(
        np.full([cells.shape[0], len(cond_names)], np.nan), columns=cond_names)
    for i, v in enumerate(cond_names):
        ds = gdal.Open('..\\Data\\Raster_Ecoregions\\' +
                       eco + '_' + v + '.tif')
        values = ds.ReadAsArray()
        factor[v] = values[cells['row'], cells['col']]
    train_index = np.arange(cells['t'].iloc[-1])
    from_gage = np.digitize(train_index, cells['t'])
    cond = factor.iloc[from_gage]
    cond = (cond - cond_mean) / cond_std

    # X
    X = np.zeros((train_index.shape[0], TIME_STEPS, INPUT_DIM))
    for i, v in enumerate(var):
        ds = gdal.Open('..\\Data\\Raster_Ecoregions\\' +
                       eco + '_' + v + '.tif')
        values = ds.ReadAsArray()[start_month + 1 - TIME_STEPS:end_month,
                                  cells['row'], cells['col']]
        add = np.tile(np.arange(0, TIME_STEPS), end_month - start_month)
        index = np.repeat(
            np.arange(0, end_month - start_month, dtype=int), TIME_STEPS)
        index = index + add
        values = values[index]
        X[:, :, i] = values.flatten('F').reshape(X.shape[0:2])
    X = (X - X_mean) / X_std

    return cond, X


def simu(model, C, eco, cells, norm):
    flow_file = '..\\Data\\Simulation\\' + eco + '.tif'
    if os.path.exists(flow_file):
        return

    y_mean, y_std = norm[-2:]
    num_months = int(cells.iloc[0, cells.columns.get_loc('num_months')])
    var = pd.read_csv(
        next(glob.iglob('..\\Data\\Basin_Time\\*.csv')), nrows=0).columns
    ds = gdal.Open('..\\Data\\Raster_Ecoregions\\' +
                   eco + '_' + var[0] + '.tif')

    # predict
    y_pred = model.predict(C, batch_size=SIMU_BATCH).squeeze()
    y_pred = y_pred * y_std + y_mean
    y_pred = y_pred.reshape(cells.shape[0], num_months)

    flow_ds = gdal.GetDriverByName('GTiff').Create(
        flow_file, ds.RasterXSize, ds.RasterYSize,
        num_months, gdal.GDT_Float64, creation)
    flow_ds.SetGeoTransform(ds.GetGeoTransform())
    flow_ds.SetProjection(ds.GetProjectionRef())

    for c in range(1, num_months + 1):
        flow_band = flow_ds.GetRasterBand(c)
        flow = np.full((ds.RasterYSize, ds.RasterXSize),
                       GRIB_NODATA, dtype=float)
        flow[cells['row'], cells['col']] = y_pred[:, c - 1]
        flow_band.SetNoDataValue(GRIB_NODATA)
        flow_band.WriteArray(flow)

    ds = None
    flow_band = None
    flow_ds = None

    return y_pred


def merge_ecos(ecoregions, out_file, no_data=None, mask=None, proj="+proj=longlat +datum=WGS84 +ellps=WGS84"):
    if os.path.exists(out_file):
        return

    tiles_path = ['..\\Data\\Simulation\\' +
                  eco + '.tif' for eco in ecoregions]
    mosaic_ds = gdal.BuildVRT('/vsimem/Mosaic.vrt', tiles_path)

    # set no data
    if mosaic_ds.GetRasterBand(1).GetNoDataValue() is not None:
        no_data = mosaic_ds.GetRasterBand(1).GetNoDataValue()
    if no_data is None:
        raise(ValueError("no_data must be initialed"))

    option = gdal.WarpOptions(creationOptions=creation, dstNodata=no_data,
                              resampleAlg=gdal.GRA_Average, srcNodata=no_data,
                              multithread=True, dstSRS=proj)
    ds = gdal.Warp(out_file, mosaic_ds, options=option)

    if mask is None:
        return

    mask_ds = treat_ras(mask)[0]
    mask_band = mask_ds.GetRasterBand(1)
    nan_mask = mask_band.ReadAsArray() == mask_band.GetNoDataValue()

    for i in range(1, 1 + ds.RasterCount):
        band = ds.GetRasterBand(i)
        if band.GetNoDataValue() is not None:
            tile_nodata = band.GetNoDataValue()
        else:
            band.SetNoDataValue(no_data)
            tile_nodata = no_data
        values = band.ReadAsArray()
        values[nan_mask] = tile_nodata
        band.WriteArray(values)


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
