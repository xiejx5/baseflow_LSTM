import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from osgeo import gdal, osr
from keras import backend as K
from tensorflow.keras.models import load_model
from cond_lstm import TIME_STEPS, INPUT_DIM, NSE


def point_value(ds, lon, lat):
    coords = lonlat2geo(ds, lon, lat)
    x, y = geo2imagexy(ds, coords[0], coords[1])
    return ds.ReadAsArray(x, y, 1, 1)


def geo2imagexy(ds, x, y):
    trans = ds.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    col, row = np.linalg.solve(a, b) - 0.5
    return int(round(col)), int(round(row))


def lonlat2geo(ds, lon, lat):
    prosrs, geosrs = getSRSPair(ds)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def getSRSPair(ds):
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(ds.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


# raster data
var = pd.read_csv(
    next(glob.iglob('..\\Data\\Basin_Time\\*.csv')), nrows=0).columns
cond_names = pd.read_excel(
    '..\\Data\\Factor_Cond.xlsx', nrows=0).columns
# lon lat
eco = 'NPL'
lon = -110.739
lat = 45.489
year = 1980
month = 1

# calculate predition year month index
ds = gdal.Open('..\\Data\\Raster_Ecoregions\\' + eco + '_' + var[0] + '.tif')
start_month = ((TIME_STEPS - 2) // 12 + 1) * 12
end_month = ds.RasterCount // 12 * 12
pred_index = (year - 1979) * 12 + month - 1
pred_band = pred_index - start_month + 1

# load normalization parameter
with open('..\\Data\\Model\\' + eco + '.pickle', 'rb') as f:
    norm = pickle.load(f)
cond_mean, cond_std, X_mean, X_std, y_mean, y_std = norm

# load point conditional parameter
factor = pd.DataFrame(
    np.full([1, len(cond_names)], np.nan), columns=cond_names)
for v in cond_names:
    ds = gdal.Open('..\\Data\\Raster_Ecoregions\\' + eco + '_' + v + '.tif')
    factor.iloc[0, factor.columns.get_loc(v)] = point_value(ds, lon, lat)
cond = factor.iloc[0]
cond = np.array((cond - cond_mean) / cond_std)

# load point X
X = np.zeros((1, TIME_STEPS, INPUT_DIM))
for i, v in enumerate(var):
    ds = gdal.Open('..\\Data\\Raster_Ecoregions\\' + eco + '_' + v + '.tif')

    values = point_value(ds, lon, lat)[
        pred_index + 1 - TIME_STEPS: pred_index + 1, :, 0]
    X[:, :, i] = values.reshape(1, 12)
X = (X - X_mean) / X_std


# set session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


# load model from h5
model = load_model('..\\Data\\Model\\' + eco + '.h5',
                   custom_objects={'NSE': [NSE]})

# simulation
C = [cond.reshape(1, 15), X]
y_pred = model.predict(C, batch_size=1).squeeze()
y_pred = y_pred * y_std + y_mean
print(f"band: {pred_band}, pred: {y_pred}")
