import os
import glob
import json
import pandas as pd
import numpy as np
from osgeo import ogr
from multiprocessing import Pool
from geotools import proj_shapefile
from point_in_polygon import work
from shapely.geometry import Point, shape
from _const import MONTH_FLOW


def records(file):
    # generator
    reader = ogr.Open(file)
    layer = reader.GetLayer(0)
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())


def in_eco(f_gages, d_shps):
    gages = pd.read_excel(f_gages, dtype={'STAID': str})
    gages['ECO'] = ''
    shps = glob.glob(os.path.join(d_shps, '*.shp'))
    proj_shps = glob.glob(os.path.dirname(
        os.path.join(d_shps, '*.shp')) + '_wgs84\\*.shp')
    points = [Point(row['LNG'], row['LAT']) for index, row in gages.iterrows()]

    for shp in shps:
        name = os.path.splitext(os.path.basename(shp))[0]
        proj_name = name + '_wgs84'
        proj_shp = os.path.join(os.path.dirname(
            shp) + '_wgs84', proj_name + ".shp")
        if proj_shp not in proj_shps:
            proj_shapefile(shp, proj_shp)
            proj_shps.append(proj_shp)

    poly_group = []
    for f in proj_shps:
        poly = records(f)
        poly_group.append([shape(feature['geometry']) for feature in poly])

    with Pool(16) as p:
        out = p.map(work, zip(points, [poly_group] * len(points)))

    eco_name = np.array([os.path.splitext(os.path.basename(f))[
        0].replace('_wgs84', '') for f in proj_shps])
    eco_index = np.array(out, dtype=int)
    gages['ECO'] = eco_name[eco_index]
    gages.to_excel(f_gages, index=False)

    for i in np.unique(eco_index):
        print(f"{eco_name[i]}: {np.sum(eco_index == i)}")
# for i, pt in enumerate(points):
#     print(i)
#     for f in proj_shps:
#         poly = records(f)
#         in_feature = (
#             pt.within(shape(feature['geometry'])) for feature in poly)
#         if any(in_feature):
#             gages.iloc[i, gages.columns.get_loc('ECO')] = (
#                 gages.iloc[i, gages.columns.get_loc('ECO')] +
#                 os.path.splitext(os.path.basename(f))[0].replace('_wgs84', ''))


def mean_flow(month, f_gages):
    month_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = pd.read_excel(f_gages, dtype={'STAID': str})
    flow_mean = []
    for i in df['STAID']:
        flow_file = MONTH_FLOW + i + '.csv'
        flow_df = pd.read_csv(flow_file)
        flow_mean.append(flow_df[flow_df['M'] == month]['B'].mean())
    df[month_str[month - 1]] = flow_mean
    df.to_excel(f_gages, index=None)
