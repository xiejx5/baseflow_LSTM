# gdal warp option
CREATION = ['TILED=YES', 'COMPRESS=DEFLATE',
            'ZLEVEL=3', 'PREDICTOR=1', 'BIGTIFF=YES']
CONFIG = ["GDAL_CACHE_MAX=128"]
GRIB_NODATA = -1.797693e+308


# Time Series begin year month
IN_BEG_YEAR = 1979
IN_BEG_MONTH = 1


# lstm hyperparameter
TIME_STEPS = 12
BATCH_SIZE = 1024
NUM_CELLS = 128
EPOCHS = 1000
COND_DIM = 12
INPUT_DIM = 8
OUTPUT_DIM = 1
DROP_P = 0.5


# baseflow begin year month
OUT_BEG_YEAR = IN_BEG_YEAR + (IN_BEG_MONTH + TIME_STEPS - 2) // 12
OUT_BEG_MONTH = (IN_BEG_MONTH + TIME_STEPS - 2) % 12 + 1


# folders
RAW_TIME = '..\\Data\\Raw\\Time'
RAW_COND = '..\\Data\\Raw\\Cond'
RAS_TIME = '..\\Data\\Raster_Time'
RAS_COND = '..\\Data\\Raster_Cond'
BSN_TIME = '..\\Data\\Basin_Time'
DAY_FLOW = '..\\Data\\Basin_DailyFlow\\'
MONTH_FLOW = '..\\Data\\Basin_MonthFlow\\'
SHP_SPLIT = '..\\Data\\Shp_Split\\'
ECOREGION = '..\\Data\\Ecoregions'
ECO_WGS = '..\\Data\\Ecoregions_wgs84'


# files
GAGE_ORG = '..\\Data\\Gages_Origin.xlsx'
GAGE = '..\\Data\\Gages.xlsx'
COND = '..\\Data\\Factor_Cond.xlsx'
