import os
import glob
import pandas as pd
from multiprocessing import freeze_support, Pool, cpu_count
from basin_work import time_series, cond_factor
from _const import (RAS_TIME, RAS_COND, GAGE, COND,
                    SHP_SPLIT, BSN_TIME)


# generate time serires data in .csv
gages = pd.read_excel(GAGE, dtype={'STAID': str})
rasters = glob.glob(os.path.join(RAS_TIME, '*.tif'))
shps = [SHP_SPLIT + i + '.shp' for i in gages['STAID']]
if not os.path.exists(BSN_TIME):
    os.mkdir(BSN_TIME)
args = zip([rasters] * len(shps), shps, [BSN_TIME] * len(shps))

if __name__ == '__main__':
    freeze_support()
    with Pool(cpu_count() - 1) as p:
        p.starmap(time_series, args)


# generate conditional basin properties in .xlsx
rasters = glob.glob(os.path.join(RAS_COND, '*.tif'))
names = [os.path.splitext(os.path.basename(i))[0] for i in rasters]
args = zip([rasters] * len(shps), shps)

if __name__ == '__main__':
    freeze_support()
    # basin condtional factors work
    with Pool(cpu_count() - 1) as p:
        output = p.starmap(cond_factor, args)

    out = pd.DataFrame(output, columns=names)
    out.to_excel(COND, index=None)
