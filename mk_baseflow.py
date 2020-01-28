import os
import glob
import gdal
import numpy as np
from mk_test import mk_test, create_tif, proj_ds, deal_calc, deal_cover
from multiprocessing import Pool, cpu_count
from calendar import monthrange

GRIB_NODATA = -1.797693e+308

# projection and reduce precipitation tif to 1979 Dec and 2018 Nov
ras = '..\\Data\\Local\\Time\\total_precipitation.tif'
start_year = 1979
start_month = 1
n_month = gdal.Open(ras).RasterCount
n_year = 39
multi_days = [monthrange(start_year + i // 12, i % 12 + 1)[1]
              for i in range(n_month)]
deal_calc(ras, '..\\Data\\Trend', multi=multi_days)
tif_prec = '..\\Data\\Trend\\total_precipitation.tif'
tif_prec_rm = os.path.join(os.path.dirname(
    tif_prec), os.path.splitext(os.path.basename(tif_prec))[0] + '_rm.tif')
if not os.path.exists(tif_prec_rm):
    gdal.Translate(tif_prec_rm, tif_prec,
                   bandList=range(12, (n_year + 1) * 12))
template_shp = next(glob.iglob('..\\Data\\Ecoregions\\*.shp'))
proj_ds(tif_prec_rm, template_shp)
tif_prec_proj = os.path.join(os.path.dirname(
    tif_prec_rm), os.path.splitext(os.path.basename(tif_prec_rm))[0] + '_proj.tif')


# projection and reduce baseflow tif to 1979 Dec and 2018 Nov
tif_base = '..\\Data\\Trend\\USA_proj.tif'
tif_base_rm = os.path.join(os.path.dirname(
    tif_base), os.path.splitext(os.path.basename(tif_base))[0] + '_rm.tif')
if not os.path.exists(tif_base_rm):
    gdal.Translate(tif_base_rm, tif_base,
                   bandList=range(1, n_year * 12 + 1))
tif_prec_final = '..\\Data\\Trend\\total_precipitation_final.tif'
deal_cover(tif_prec_final, tif_base_rm, tif_prec_proj)


# baseflow mk
ds = gdal.Open(tif_base_rm)
im = ds.ReadAsArray()
im = np.transpose(im, [1, 2, 0])
im = np.ma.masked_array(
    im, mask=im == ds.GetRasterBand(1).GetNoDataValue())

quar_im = []
for i in range(4):
    quar_add = np.tile(np.arange(i * 3, i * 3 + 3),
                       ds.RasterCount // 12)
    quar_idx = quar_add + np.repeat(np.arange(0, ds.RasterCount, 12), 3)
    quar_idx = quar_idx.reshape(-1, 3)
    quar_3d = im[:, :, quar_idx].sum(axis=-1)
    quar_im.append(quar_3d.reshape(-1, quar_3d.shape[-1]))
comb_base = np.ma.concatenate(quar_im, axis=0)

result = np.zeros((comb_base.shape[0], 1))
valid_idx = np.where(np.any(comb_base.mask, axis=1) == False)[0]

if __name__ == '__main__':
    with Pool(cpu_count() - 1) as p:
        result[valid_idx, 0] = p.map(mk_test, comb_base[valid_idx])

quar_mk = [np.ma.masked_array(
    i, mask=np.any(quar_im[0].mask, axis=1)
    [:, np.newaxis]).reshape(im.shape[:2])
    for i in np.split(result, 4)]

comb_mk = np.ma.dstack(quar_mk)
comb_mk.set_fill_value(GRIB_NODATA)
create_tif('MK_Baseflow.tif', ds, comb_mk.filled())


# precipitation mk
ds = gdal.Open(tif_prec_final)
im = ds.ReadAsArray()
im = np.transpose(im, [1, 2, 0])
im = np.ma.masked_array(
    im, mask=im == ds.GetRasterBand(1).GetNoDataValue())

quar_im = []
for i in range(4):
    quar_add = np.tile(np.arange(i * 3, i * 3 + 3),
                       ds.RasterCount // 12)
    quar_idx = quar_add + np.repeat(np.arange(0, ds.RasterCount, 12), 3)
    quar_idx = quar_idx.reshape(-1, 3)
    quar_3d = im[:, :, quar_idx].sum(axis=-1)
    quar_im.append(quar_3d.reshape(-1, quar_3d.shape[-1]))
comb_prec = np.ma.concatenate(quar_im, axis=0)

result = np.zeros((comb_prec.shape[0], 1))
valid_idx = np.where(np.any(comb_prec.mask, axis=1) == False)[0]

if __name__ == '__main__':
    with Pool(cpu_count() - 1) as p:
        result[valid_idx, 0] = p.map(mk_test, comb_prec[valid_idx])

quar_mk = [np.ma.masked_array(
    i, mask=np.any(quar_im[0].mask, axis=1)
    [:, np.newaxis]).reshape(im.shape[:2])
    for i in np.split(result, 4)]

comb_mk = np.ma.dstack(quar_mk)
comb_mk.set_fill_value(GRIB_NODATA)
create_tif('MK_Precipitation.tif', ds, comb_mk.filled())


# baseflow / precipitation mk
ds = gdal.Open(tif_base_rm)
im = ds.ReadAsArray()
im = np.transpose(im, [1, 2, 0])
im = np.ma.masked_array(
    im, mask=im == ds.GetRasterBand(1).GetNoDataValue())

quar_im = []
for i in range(4):
    quar_add = np.tile(np.arange(i * 3, i * 3 + 3),
                       ds.RasterCount // 12)
    quar_idx = quar_add + np.repeat(np.arange(0, ds.RasterCount, 12), 3)
    quar_idx = quar_idx.reshape(-1, 3)
    quar_3d = im[:, :, quar_idx].sum(axis=-1)
    quar_im.append(quar_3d.reshape(-1, quar_3d.shape[-1]))
comb_prec = comb_base / comb_prec

result = np.zeros((comb_prec.shape[0], 1))
valid_idx = np.where(np.any(comb_prec.mask, axis=1) == False)[0]

if __name__ == '__main__':
    with Pool(cpu_count() - 1) as p:
        result[valid_idx, 0] = p.map(mk_test, comb_prec[valid_idx])

quar_mk = [np.ma.masked_array(
    i, mask=np.any(quar_im[0].mask, axis=1)
    [:, np.newaxis]).reshape(im.shape[:2])
    for i in np.split(result, 4)]

comb_mk = np.ma.dstack(quar_mk)
comb_mk.set_fill_value(GRIB_NODATA)
create_tif('MK_Ratio.tif', ds, comb_mk.filled())
