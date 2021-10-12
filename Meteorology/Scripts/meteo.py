import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys
import funcmeteo as f2


##This code creates plots of meteorological (rainfall and wind speed) data

###############################
###Open the meteorology table
meteo = pd.read_csv('../Entrance/tabmeteorology.csv')
# print(meteo)
# print(meteo.describe())


###Edit meteorology table
#Convert datetime15 to datetime format
meteo.loc[:,'datetime1'] = pd.to_datetime(meteo.datetime15, format='%Y/%m/%d %H:%M:%S')

#Select rainfall and wind speed until Nov30, 2019 (there is data until Dec 4)
raws = meteo.loc[meteo.datetime1.between('2014-10-02 00:00:00','2019-11-30 23:45:00'),['datetime1','ra','wsmx']]
# print (raws)


#Plot four subplots together
plt.rc('font', family='Times New Roman', size=12)
fig, (ax3, ax4, ax1, ax2) = plt.subplots(4,1, figsize=(11, 12))
# fig, ax  = plt.subplots(1,1, figsize=(11, 10))

f2.functionrainfall(raws, ax1)
f2.functionwind(raws, ax2)
f2.functionrainfall1(raws, ax3)
f2.functionwind1(raws, ax4)

ax3.set_title('(a)', x=0.025, y=0.85) 
ax4.set_title('(b)', x=0.025, y=0.85)
ax1.set_title('(c)', x=0.025, y=0.85)
ax2.set_title('(d)', x=0.025, y=0.85)




# f2.functionsoilh(sh, ax3)
# plt.yscale('log')
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('../Exit/gap_4subplots.png', dpi=300, bbox_inches='tight')
plt.close()


###Bars of monthly rainfall

rainm = f2.functionrainfallmonth(raws)
print(rainm.dtypes)

rainm.loc[:,'month2'] = pd.to_datetime(rainm.month, format='%Y-%m')
print(rainm)

rainm['justmonth'] = rainm['month2'].dt.month
print(rainm)

# Mean of rainfall for each month
tabdin = pd.pivot_table(rainm, index='justmonth', values='ra', aggfunc=np.mean)
print(tabdin)

plt.figure(figsize=(11, 5))
plt.rc('font', family='Times New Roman', size=16)

plt.bar(tabdin.index, tabdin.ra)

plt.xticks(tabdin.index, ('J','F','M','A','M','J','J','A','S','O','N','D'))
plt.xlabel('Months', labelpad=10, fontsize=16)
plt.ylabel('Rainfall (mm)', labelpad=15, fontsize=16)
plt.savefig('../Exit/monthly_rainfall.png', dpi=300, bbox_inches='tight')
plt.close()

