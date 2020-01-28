import os
import gdal
import geotools as gt
import numpy as np
from multiprocessing import Pool, cpu_count


def band_multiply(ds, band_idx, multi, out_path):
    if isinstance(ds, str):
        ras = ds
        ds = gdal.Open(ras)
    else:
        ras = ds.GetDescription()

    ext = os.path.splitext(os.path.basename(out_path))[1]
    if ext != '.tif':
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, os.path.splitext(
            os.path.basename(ras))[0] + '_' + str(band_idx) + '.tif')
    else:
        out_file = os.path.join(os.path.dirname(out_path),
                                os.path.splitext(os.path.basename(out_path))[0]
                                + '_' + str(band_idx) + '.tif')
    if os.path.exists(out_file):
        return

    input_args = {'A': ras, 'A_band': band_idx}
    calc_arg = f'A*{float(multi)}'
    gt.Calc(calc_arg, out_file, creation_options=gt.CREATION,
            quiet=True, **input_args)


def multiply_list(ds, multi_list, out_path):
    if isinstance(ds, str):
        ras = ds
        ds = gdal.Open(ras)
    else:
        ras = ds.GetDescription()

    n_band = ds.RasterCount
    if len(multi_list) != n_band:
        raise('multi list length not equal to band counts')

    ext = os.path.splitext(os.path.basename(out_path))[1]
    if ext != '.tif':
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, os.path.splitext(
            os.path.basename(ras))[0] + '.tif')
    else:
        out_file = out_path

    if os.path.exists(out_file):
        return

    band_idx_list = np.arange(1, n_band + 1, dtype=np.int)
    args = zip([ras] * n_band, band_idx_list,
               multi_list, [out_path] * n_band)
    with Pool(min(cpu_count() - 1), n_band) as p:
        p.starmap(band_multiply, args)
