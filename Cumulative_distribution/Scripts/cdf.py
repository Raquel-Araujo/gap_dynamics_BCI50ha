import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
# from scipy import stats
import math


##Gaps with centroids inside the plot
gapsarea=pd.read_csv("../Entrance/gaps_2014_2019_centroid.csv")

##Remove the long interval
# ~ means is not in the list
longint = ['2016-05-18']
gapsarea = gapsarea[~gapsarea['date'].isin(longint)]

#Proportions
# x = [5,10,25,50,100,200,400]
x= np.linspace(gapsarea.area_m2.min(), gapsarea.area_m2.max(), 1000)
coletornum = []
coletorarea = []

i = 0

while i < len(x):

    gaps = gapsarea.loc[gapsarea.area_m2 < x[i]]
    n = len(gaps)
    porcnum = float(len(gaps))/float(len(gapsarea))*100
    porcarea = (gaps.area_m2.sum()/gapsarea.area_m2.sum())*100
    coletornum.append(porcnum)
    coletorarea.append(porcarea)

    i+=1



plt.figure(figsize=(4, 3))
plt.rc('font', family='Times New Roman', size=12)


plt.plot(x,coletorarea, "k" , label = "Area", linewidth=0.8)
plt.plot(x,coletornum, "grey" , label = "Number", linewidth=0.8)
plt.xscale('log')
plt.legend()

plt.xticks([2,5,10,20,50,100,200,500],[2,5,10,20,50,100,200,500])
plt.ylabel('Cumulative distribution (%)', labelpad=10)
plt.savefig('../Exit/cumulative_percentage.png', dpi=300, bbox_inches='tight')
plt.close()



sumgapsx = 0.5*gapsarea.area_m2.sum()

x1 = np.array(x)
cumularea = np.array(coletorarea)

df = pd.DataFrame(np.array([x1, cumularea]).T, columns=('area', 'cumulative'))

##Area of cumulative distribution = 50% is 87.1 m2
a = df.loc[df.cumulative>=50,:]
a.round(1).to_csv('../Exit/cumulative_distribution_50%.csv')


