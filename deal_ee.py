import os
import ee


os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
ee.Initialize()

username = 'xiejx5'
ID_field = "geeID"
fc_file = '1633-na'
scalePix = 30
folder = 'basin_properties'

# load pts or poly file
srtm = ee.Image('USGS/SRTMGL1_003')
fc = ee.FeatureCollection('users/' + username + '/' + str(fc_file))


fc_mean = srtm.reduceRegions(collection=fc,
                             reducer=ee.Reducer.mean(),
                             scale=scalePix)
task = ee.batch.Export.table.toDrive(collection=fc_mean
                                     .filter(ee.Filter.neq('mean', None))
                                     .select(['.*'], newProperties=None,
                                             retainGeometry=False),
                                     description='dem',
                                     folder=folder,
                                     fileFormat='CSV')
task.start()
