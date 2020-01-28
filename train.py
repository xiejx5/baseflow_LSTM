import os
import glob
from train_work import train_work
from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import freeze_support, Pool


# Prepare Ecoregions
if not os.path.exists("..\\Data\\Model\\"):
    os.mkdir("..\\Data\\Model\\")
shps = glob.glob('..\\Data\\Ecoregions_wgs84\\*.shp')
ecos = [os.path.splitext(os.path.basename(f))[
    0].replace('_wgs84', '') for f in shps]

with ThreadPoolExecutor(max_workers=len(ecos)) as p:
    output = p.map(train_work, ecos)
