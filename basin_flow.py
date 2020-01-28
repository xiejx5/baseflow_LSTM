import os
import numpy as np
import pandas as pd
from calendar import monthrange
from collections import OrderedDict
from basin_eco import in_eco, mean_flow
from _const import (OUT_BEG_YEAR, OUT_BEG_MONTH, DAY_FLOW,
                    MONTH_FLOW, ECOREGION, GAGE_ORG, GAGE)

gages = pd.read_excel(GAGE_ORG, dtype={'STAID': str})
drop_index = []

for index, g in gages.iterrows():
    f_in = DAY_FLOW + g['STAID'] + '.txt'
    f_out = MONTH_FLOW + g['STAID'] + '.csv'
    if os.path.exists(f_out):
        continue

    df = pd.read_csv(f_in, sep='\t', header=None, names=['Y', 'M', 'D', 'B'])
    year = np.unique(df['Y'])
    year = year[year >= OUT_BEG_YEAR]
    year = np.repeat(year, 12)
    month = np.arange(1, 13)
    month = np.tile(month, int(year.shape[0] / 12))
    first_year_delete = np.where(
        (month < OUT_BEG_MONTH) & (year == OUT_BEG_YEAR))[0]
    year = np.delete(year, first_year_delete)
    month = np.delete(month, first_year_delete)
    flow = np.full(month.shape, np.nan)
    for i, (y, m) in enumerate(zip(year, month)):
        select = (df['Y'] == y) & (df['M'] == m)
        month_days = monthrange(y, m)[1]
        if np.sum(select) == month_days:
            # convert unit from m3/s to mm/day
            flow[i] = np.sum(df['B'][select]) * 86.4 / (g['Area'] * month_days)

    delete_rows = np.where(np.isnan(flow))
    year = np.delete(year, delete_rows)
    month = np.delete(month, delete_rows)
    flow = np.delete(flow, delete_rows)

    if flow.shape[0] < 10:
        drop_index.append(index)
        continue

    if not os.path.exists(MONTH_FLOW):
        os.mkdir(MONTH_FLOW)
    df_out = pd.DataFrame.from_dict(
        OrderedDict(zip(['Y', 'M', 'B'], [year, month, flow])))
    df_out.to_csv(f_out, index=False)

gages = gages.drop(drop_index)
gages.to_excel(GAGE, index=None)
in_eco(GAGE, ECOREGION)
mean_flow(4, GAGE)
