import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import funcoes as f
import math
# import matplotlib.ticker as mtick


heightdrop = pd.read_csv('../Entrance/gaps1419_centroid_join_heightdrop.csv')

heightdrop.rename(columns = {'MEAN':'heightdrop'}, inplace=True)

#Remove long interval
# longint = ['2016-05-18']
longint = ['18/05/16']

heightdropf = heightdrop[~heightdrop['date'].isin(longint)]

# Height drop - filter negative values (the positive values are model errors)
dropneg = heightdropf.loc[heightdropf.heightdrop < 0]
# 61 were removed


# Convert the drop values into positive for graphing
dropneg['heightdropmod'] = dropneg.heightdrop*(-1)

#Classes area
binsarea = [2,5,10,20,50,100,200,500]
areamin = [2,5,10,20,50,100,200]

areamax = np.concatenate((areamin[1:],[500])) ###remove the first and add 500

width = areamax - areamin


#Classes height drop
binsheightdrop = [0,2,5,10,30]
heightdropmin = [0,2,5,10]

dropneg.loc[:,('classearea')] = pd.cut(dropneg['area_m2'], binsarea, labels=areamin)
dropneg.loc[:,('classeheightdrop')] = pd.cut(dropneg['heightdropmod'], binsheightdrop, labels=heightdropmin)

dropnegsel = dropneg.loc[:,['heightdropmod','classeheightdrop', 'classearea']]

freq = pd.pivot_table(dropnegsel, values = 'heightdropmod', index="classearea", columns = 'classeheightdrop', aggfunc=len, fill_value=0)
freq.columns = ['0', '2', '5', '>10']
freq.to_csv('../Exit/pivot_table.csv')

freq1 = (freq/len(dropnegsel))*100
freq1.reset_index(inplace=True)


plt.figure(figsize=(4, 3))
plt.rc('font', family='Times New Roman', size=12)

freq1.plot(x='classearea', kind='bar', stacked=True, color=['gainsboro', 'silver', 'dimgrey', 'k'], align='edge', rot=0, width=1.0,
            figsize=(4, 3), fontsize=12)
plt.xlabel(r'Canopy disturbance area (m$^{2}$)', labelpad=10)
plt.ylabel('Frequency (%)', labelpad=10, fontsize=12)
plt.legend(['0-2 m', '2-5 m', '5-10 m', '>10 m'], title="Height drop", fontsize=10)
plt.yticks([0,10,20,30,40], [0,10,20,30,40], fontsize=12)
plt.xticks([0,1,2,3,4,5,6,7],['2','5','10','20','50','100', '200', '500'], fontsize=12)


plt.savefig('../Exit/stackbar_area_heightdrop.png', dpi=300, bbox_inches='tight')
plt.close()



# # print(width)

# ###Plot log scale
# plt.figure(figsize=(4, 3))
# plt.rc('font', family='Times New Roman', size=12)

# # Heights of bars1 + bars2
# bars = np.add(freq1['0'], freq1['2']).tolist()
# five = freq1['5'].tolist()
# bars1 = np.add(bars, five)
# # print(bars)
# # print(bars1)

# ###Plot x axis log scale
# plt.figure(figsize=(4, 3))
# plt.rc('font', family='Times New Roman', size=12)

# # Heights of bars1 + bars2
# bars = np.add(freq1['0'], freq1['2']).tolist()
# five = freq1['5'].tolist()
# bars1 = np.add(bars, five)
# # print(bars)
# # print(bars1)

# width = 1.0

# print(np.array((freq1.classearea), dtype=str))



# # Create brown bars
# plt.bar(np.array((freq1.classearea), dtype=str), freq1['0'], color = 'white', edgecolor='k', width=width,align='edge', label='0-2 m')
# # Create green bars (middle), on top of the first ones
# plt.bar(np.array((freq1.classearea), dtype=str), freq1['2'], bottom=freq1['0'], color='lightgray', edgecolor='k', width=width, align='edge', label='2-5 m')
# # Create green bars (top)
# plt.bar(np.array((freq1.classearea), dtype=str), freq1['5'], bottom=bars, color='darkgray', edgecolor='k', width=width, align='edge', label='5-10 m')
# plt.bar(np.array((freq1.classearea), dtype=str), freq1['>10'], bottom=bars1, color='k', edgecolor='k', width=width, align='edge', label='>10 m')

# # plt.xscale('log')
# # plt.yscale('log')

# plt.xlabel(r'Canopy disturbance area (m$^{2}$)', labelpad=10)
# plt.ylabel('Frequency (%)', labelpad=10)

# # plt.xticks([2,5,10,20,50,100,200,500], [2,5,10,20,50,100,200,500])
# # plt.yticks([1,10,100], [1,10,100])
# plt.minorticks_off()

# plt.legend(title="Height drop", loc='upper left', prop={'size': 8})

# plt.savefig('../Exit/stackbar_area_heightdrop1_logx.png', dpi=300, bbox_inches='tight')

# plt.close()
