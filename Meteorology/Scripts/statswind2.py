import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys

wind = pd.read_csv('../Exit/wind_max_1day.csv')
print(wind)

#Convert wind to m/s
wind['wsmx_ms'] = wind.wsmx/3.6
print(wind)

wind.loc[:,'date1d'] = pd.to_datetime(wind.datetime1d, format='%Y-%m-%d')
wind['month'] = wind['date1d'].dt.month

#Multiply by 1 transforms True/False into 0 and 1
# 1 = wet (May to December)
# 0 = dry (January to April)
wind['season'] = (wind.month>=5)*1
#Drop the last row
wind = wind[:-1]
print(wind)

#Group by season and do the mean of the wind speed values
windmean = wind.groupby(wind.season).wsmx_ms.mean()
windmean.to_csv('../Exit/mean_1day_wsmx_ms_seasons.csv')

# #Dry season

# startdry = '01-01'
# enddry = '04-30'




