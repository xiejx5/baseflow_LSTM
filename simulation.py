import os
import glob
import pickle
from cond_lstm import NSE
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model
from simulation_work import partition_work
from multiprocessing import freeze_support, Pool
from simulation_data import (get_info_simu, load_data_simu,
                             simu, merge_ecos, proj_ds)


# region shape
shps = glob.glob('..\\Data\\Ecoregions_wgs84\\*.shp')
ecoregions = [os.path.splitext(os.path.basename(f))[
    0].replace('_wgs84', '') for f in shps]


# clip rasters with regions
if not os.path.exists('..\\Data\\Raster_Ecoregions'):
    os.makedirs('..\\Data\\Raster_Ecoregions')

if __name__ == '__main__':
    freeze_support()
    with Pool(3) as p:
        p.map(partition_work, shps)

# prepare folder
if not os.path.exists('..\\Data\\Simulation'):
    os.mkdir('..\\Data\\Simulation')


# set session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

for eco in ecoregions:
    flow_file = '..\\Data\\Simulation\\' + eco + '.tif'
    if os.path.exists(flow_file):
        continue

    # load normalization parameter
    with open('..\\Data\\Model\\' + eco + '.pickle', 'rb') as f:
        norm = pickle.load(f)

    # load data for simulation
    cells, start_month, end_month = get_info_simu(eco)
    cond, X = load_data_simu(eco, cells, start_month, end_month, norm)
    C = [cond, X]

    # load model from h5
    model = load_model('..\\Data\\Model\\' + eco + '.h5',
                       custom_objects={'NSE': [NSE]})

    # simulation
    simu(model, C, eco, cells, norm)


# merge all ecoregions
out_file = '..\\Data\\Simulation\\USA.tif'
merge_ecos(ecoregions, out_file, mask='..\\Data\\Local\\Cond\\CLAY.tif')

# convert to projection coordination
template_shp = next(glob.iglob('..\\Data\\Ecoregions\\*.shp'))
proj_ds(out_file, template_shp)
month_str = '0' + str(IN_BEG_MONTH) if IN_BEG_MONTH < 10 else str(IN_BEG_MONTH)
time_beg = np.datetime64(f'{IN_BEG_YEAR}-{month_str}')
time_arr = np.arange(time_beg, time_beg + np.timedelta64(ds.RasterCount, 'M'),
                     np.timedelta64(1, 'M'))
month_days = ((time_arr + np.timedelta64(1, 'M')).astype('datetime64[D]')
              - time_arr.astype('datetime64[D]')).astype(int)
# proj_ds(out_file, template_shp, clip_shp='..\\Data\\Shp_Map\\usa_wgs84.shp')

# close sess
sess.close()
