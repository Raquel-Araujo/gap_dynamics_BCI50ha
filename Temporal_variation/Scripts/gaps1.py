import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
# import function as f
import sys

##This code creates figures of temporal variaton of canopy disturbances (Fig 3 and S5)


gapsdays = pd.read_csv('../Entrance/gaps_area_days_5years.csv')
gapsdaysnumber = pd.read_csv('../Entrance/gaps_number_days_5years.csv')

###Plot bars area
#Convert days in integer format to datetime format. Map atributes to each value of the column
gapsdays['dayscor'] = gapsdays['days'].map(dt.timedelta).astype('timedelta64[D]')
gapsdaysnumber['dayscor'] = gapsdaysnumber['days'].map(dt.timedelta).astype('timedelta64[D]')

#Convert date to datetime format
gapsdays.loc[:,'date1'] = pd.to_datetime(gapsdays.date, format='%Y/%m/%d')
gapsdaysnumber.loc[:,'date1'] = pd.to_datetime(gapsdaysnumber.date, format='%Y/%m/%d')

###Create column of month

gapsdays['month'] = gapsdays['date1'].dt.strftime('%Y-%m')
gapsdaysnumber['month'] = gapsdaysnumber['date1'].dt.strftime('%Y-%m')


####Figure area

colors = [
[214,133,137],
]
cor = list(np.array(colors)/255.)

plt.figure(figsize=(11, 5))
plt.rc('font', family='Times New Roman', size=14)


inicio = [
'2014-05-01',
'2015-05-01',
'2016-05-01',
'2017-05-01',
'2018-05-01',
'2019-05-01',
]

final = [
'2014-12-31',
'2015-12-31',
'2016-12-31',
'2017-12-31',
'2018-12-31',
'2019-12-31',
]

i = 0
while i< len(inicio):

    plt.axvspan(pd.to_datetime(inicio[i]), pd.to_datetime(final[i]), alpha=0.5, color='gray')

    i+=1


plt.bar(gapsdays['date1'], gapsdays['areapercmonth'], width=-gapsdays['dayscor'], align='edge', edgecolor='k', color=cor)

plt.xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))
plt.ylabel(r'Canopy disturbance rate (% mo$^{-1}$)', labelpad=15, fontsize=16)
plt.xlabel('Images dates', labelpad=10, fontsize=16)


plt.savefig('../Exit/gap_area_5y_bars.png', dpi=300, bbox_inches='tight')
plt.close()


####Figure number

colors = [
[214,133,137],
]
cor = list(np.array(colors)/255.)

plt.figure(figsize=(11, 5))
plt.rc('font', family='Times New Roman', size=14)


inicio = [
'2014-05-01',
'2015-05-01',
'2016-05-01',
'2017-05-01',
'2018-05-01',
'2019-05-01',
]

final = [
'2014-12-31',
'2015-12-31',
'2016-12-31',
'2017-12-31',
'2018-12-31',
'2019-12-31',
]

i = 0
while i< len(inicio):

    plt.axvspan(pd.to_datetime(inicio[i]), pd.to_datetime(final[i]), alpha=0.5, color='gray')

    i+=1


plt.bar(gapsdaysnumber['date1'], gapsdaysnumber['numbermonthha'], width=-gapsdaysnumber['dayscor'], align='edge', edgecolor='k', color=cor)

plt.xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))
plt.ylabel(r'Number of disturbances (n ha$^{-1}$ mo$^{-1}$)', labelpad=15, fontsize=16)
plt.xlabel('Images dates', labelpad=10, fontsize=16)


plt.savefig('../Exit/gap_number_5y_bars.png', dpi=300, bbox_inches='tight')
plt.close()

