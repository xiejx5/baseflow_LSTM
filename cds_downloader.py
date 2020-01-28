import os
import cdsapi
from concurrent.futures import ThreadPoolExecutor
from _const import RAW_COND, RAW_TIME

if not os.path.exists(RAW_COND):
    os.makedirs(RAW_COND)
if not os.path.exists(RAW_TIME):
    os.makedirs(RAW_TIME)


def cds_down(v):
    out_file = os.path.join(RAW_TIME, v + '.grib')
    if os.path.exists(out_file):
        return

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': v,
            'year': [
                '1979', '1980', '1981',
                '1982', '1983', '1984',
                '1985', '1986', '1987',
                '1988', '1989', '1990',
                '1991', '1992', '1993',
                '1994', '1995', '1996',
                '1997', '1998', '1999',
                '2000', '2001', '2002',
                '2003', '2004', '2005',
                '2006', '2007', '2008',
                '2009', '2010', '2011',
                '2012', '2013', '2014',
                '2015', '2016', '2017',
                '2018', '2019'
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12'
            ],
            'time': '00:00',
            'format': 'grib'
        },
        out_file)


variable = [
    '10m_wind_speed', '2m_temperature', 'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation', 'snowfall', 'surface_net_solar_radiation',
    'total_precipitation', 'potential_evaporation', 'snowmelt', 'surface_net_thermal_radiation'
]
with ThreadPoolExecutor(max_workers=len(variable)) as executor:
    executor.map(cds_down, variable)


# Non-time series
variable = ['high_vegetation_cover', 'low_vegetation_cover']
for v in variable:
    out_file = os.path.join(RAW_TIME, v + '.grib')
    if not os.path.exists(out_file):
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'format': 'grib',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': v,
                'year': '2019',
                'month': '06',
                'time': '00:00'
            },
            out_file)
