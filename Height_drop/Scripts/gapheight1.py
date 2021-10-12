import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
from pygam import LinearGAM, s, f
import statsmodels.api as sm



heightdrop = pd.read_csv('../Entrance/gaps1419_centroid_join_heightdrop.csv')

heightdrop.rename(columns = {'MEAN':'heightdrop'}, inplace=True)

#Remove long interval
# longint = ['2016-05-18']
longint = ['18/05/16']
heightdropf = heightdrop[~heightdrop['date'].isin(longint)]


# Height drop - filter negative values (the positive values are model errors)
# 61 observations excluded
dropneg = heightdropf.loc[heightdropf.heightdrop < 0]

# Convert the drop values into positive for graphing
dropneg['heightdropmod'] = dropneg.heightdrop*(-1)


##GAM
x = np.log(dropneg.area_m2)
y = np.log(dropneg.heightdropmod)


X = sm.add_constant(x)
gam = LinearGAM(s(0) + s(1)).gridsearch(X.values, y.values)

sumario = gam.summary()
print(sumario)
estatistica = list(gam.statistics_.keys())
print(estatistica)


x1 = np.linspace(x.min(), x.max(), 100)

X1 = sm.add_constant(x1)
ypred = gam.predict(X1)


# Scatter - values description x log scale
plt.figure(figsize=(4, 3))
plt.rc('font', family='Times New Roman', size=12)

plt.scatter(dropneg.area_m2,dropneg.heightdropmod, edgecolors='none', facecolors='k', s=20, alpha=0.3, label=None)
plt.plot(np.exp(x1), np.exp(ypred), 'r--', label='GAM')

# plt.xlabel(r'Canopy disturbance area (m$^{2}$)', labelpad=10)
plt.ylabel('Height drop (m)', labelpad=10)
plt.xscale('log')

plt.xticks([2,5,10,20,50,100,200,500],[2,5,10,20,50,100,200,500])
plt.legend(loc='best')
plt.savefig('../Exit/gap_area_heighdrop_gam.png', dpi=300, bbox_inches='tight')
plt.close()