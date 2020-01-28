import os
import glob
import pandas as pd
import numpy as np
from cond_lstm import get_info
from cross_work import parallel_val
from sklearn.model_selection import StratifiedKFold
from multiprocessing import freeze_support, Pool

# Prepare Ecoregions
if not os.path.exists("..\\Data\\Model\\"):
    os.mkdir("..\\Data\\Model\\")
shps = glob.glob('..\\Data\\Ecoregions_wgs84\\*.shp')
ecoregions = [os.path.splitext(os.path.basename(f))[
    0].replace('_wgs84', '') for f in shps]

# Prepare names
if not os.path.exists("..\\Data\\Model\\"):
    os.mkdir("..\\Data\\Model\\")


if __name__ == '__main__':
    freeze_support()

    for eco in ecoregions:
        if os.path.exists('..\\Data\\Model\\' + eco + '.xlsx'):
            continue

        fold = 10

        # k-fold
        gages, factor, flow = get_info(1979, 1, eco)
        n_samples = flow.shape[0]
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)

        output = []
        splits = list(enumerate(skf.split(np.zeros(n_samples), flow['C'])))
        args = list(zip(splits, [[gages, factor, flow]] * fold))
        with Pool(fold) as p:
            output = p.map(parallel_val, args)

        cross = pd.DataFrame(output, columns=['index', eco, eco + '_N'])
        cross.to_excel('..\\Data\\Model\\' + eco + '.xlsx', index=None)
