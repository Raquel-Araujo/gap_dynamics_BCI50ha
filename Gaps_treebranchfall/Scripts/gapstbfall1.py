import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import function as f
import function1 as f1
import funcfit as ffit
import sys
import math

#Only 863 classified gaps (150 long interval gaps and 35 undetermined gaps are not part of this table)
gaps = pd.read_csv('../Entrance/gaps_classes_20141126_20191128.csv')
print(gaps)

areasum = np.sum(gaps.area_m2)
pd.Series(areasum).round(1).to_csv('../Exit/sum_total_area.csv')


#Each class
treef = gaps.loc[gaps.treefall==1]
branchf = gaps.loc[gaps.branchfall==1]
stand = gaps.loc[gaps.stand_dead==1]

treef.describe().to_csv('../Exit/treefall_describe.csv')
branchf.describe().to_csv('../Exit/branchfall_describe.csv')
stand.describe().to_csv('../Exit/standing_describe.csv')


###Area
areatreef = np.sum(treef.area_m2)
areabranchf = np.sum(branchf.area_m2)
areastand = np.sum(stand.area_m2)

areatreeporc = (areatreef/(areatreef+areabranchf+areastand))*100
areabranchporc = (areabranchf/(areatreef+areabranchf+areastand))*100
areastandporc = (areastand/(areatreef+areabranchf+areastand))*100

summary = pd.DataFrame([areatreeporc, areabranchporc, areastandporc], index=['treefall','branchfall','stand_dead'])
summary.to_csv('../Exit/summary_classes_area.csv')

###Area proportions counting only treefall and branchfall
areatreeporc = (areatreef/(areatreef+areabranchf))*100
areabranchporc = (areabranchf/(areatreef+areabranchf))*100

summarytb = pd.DataFrame([areatreeporc, areabranchporc], index=['treefall','branchfall'])
summarytb.to_csv('../Exit/summary_classes_area_treebranchfall.csv')

###Number
ntreef = len(treef)
nbranchf = len(branchf)
nstand = len(stand)

ntreeporc = (ntreef*1./(ntreef+nbranchf+nstand)*1.)*100
nbranchporc = (nbranchf*1./(ntreef+nbranchf+nstand)*1.)*100
nstandporc = (nstand*1./(ntreef+nbranchf+nstand)*1.)*100

summary = pd.DataFrame([ntreeporc, nbranchporc, nstandporc], index=['treefall','branchfall','stand_dead'])
summary.to_csv('../Exit/summary_classes_number.csv')

###Number proportions counting only treefall and branchfall
ntreeporc = (ntreef*1./(ntreef+nbranchf)*1.)*100
nbranchporc = (nbranchf*1./(ntreef+nbranchf)*1.)*100

summarytb = pd.DataFrame([ntreeporc, nbranchporc], index=['treefall','branchfall'])
summarytb.to_csv('../Exit/summary_classes_number_treebranchfall.csv')



####################################################################
###Plot bars treefall and branchfall


#Plot all together
plt.rc('font', family='Times New Roman', size=12)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(11, 8))


f1.funcdaysarea(treef, 'Treefall', 1, ax1)
f1.funcdaysarea(branchf, 'Branchfall', 0, ax2)
f1.funcdaysnumber(treef, 'Treefall', 1, ax3)
f1.funcdaysnumber(branchf, 'Branchfall', 0, ax4)

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('../Exit/gap_4subplots_ha.png', dpi=300, bbox_inches='tight')
plt.close()


###Scatter plots
areatreefall = f.funcdaysarea(treef, 'Treefall')
areabranchfall = f.funcdaysarea(branchf, 'Branchfall')
numbertreefall = f.funcdaysnumber(treef, 'Treefall')
numberbranchfall = f.funcdaysnumber(branchf, 'Branchfall')


tabelas = [areatreefall, areabranchfall, numbertreefall, numberbranchfall]
for i in tabelas:
    i['year'] = i['date1'].dt.year


#########################
###Two scatter plots together log scale

#Fit loglog for plot purposes
x = np.log(areatreefall.areapercmonth)
y = np.log(areabranchfall.areapercmonth)

sumariolog = ffit.linear(x, y, "area_log")

xn = np.log(numbertreefall.numbermonthha)
yn = np.log(numberbranchfall.numbermonthha)

sumariolognumber = ffit.linear(xn, yn, "number_log")


####Scatter plot

##Lines 1:1
#These are limits of the plot
xplot = np.linspace(0.005, 0.9, 100)
yplot = 1.*xplot

xnplot = np.linspace(0.005, 0.9, 100)
ynplot = 1.*xnplot


plt.rc('font', family='Times New Roman', size=12)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 3.5))

ax1.scatter(areatreefall['areapercmonth'],areabranchfall['areapercmonth'], edgecolors='none', facecolors='k', s=20, alpha=0.5, label=None)
ax2.scatter(numbertreefall['numbermonthha'],numberbranchfall['numbermonthha'], edgecolors='none', facecolors='k', s=20, alpha=0.5, label=None)

ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xscale('log')

ax1.plot(xplot, yplot, color='k', linestyle='-', linewidth=0.5, label='1:1')
ax2.plot(xnplot, ynplot, color='k', linestyle='-', linewidth=0.5, label='1:1')

ax1.legend(loc='lower right')
ax2.legend(loc='lower right')

ax1.set_xlabel(r'Area treefall (% mo$^{-1}$)', fontsize=12, labelpad=10)
ax1.set_ylabel(r'Area branchfall (% mo$^{-1}$)', fontsize=12)
ax2.set_xlabel(r'Number treefall (n ha$^{-1}$ mo$^{-1}$)', fontsize=12, labelpad=10)
ax2.set_ylabel(r'Number branchfall (n ha$^{-1}$ mo$^{-1}$)', fontsize=12)

ax1.text(0.01,0.65,'r = %0.2f' % sumariolog.loc['r'])
ax2.text(0.01,0.65,'r = %0.2f' % sumariolognumber.loc['r'])

ax1.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
ax1.set_xticklabels(['0.01', '0.02', '0.05','0.1','0.2', '0.5'])
ax1.set_yticks([0.01, 0.1, 0.5])
ax1.set_yticklabels(['0.01', '0.1', '0.5'])

ax2.set_xticks([0.01, 0.02,0.05, 0.1, 0.2, 0.5])
ax2.set_xticklabels(['0.01','0.02','0.05', '0.1', '0.2', '0.5'])
ax2.set_yticks([0.01, 0.1, 0.5])
ax2.set_yticklabels(['0.01', '0.1', '0.5'])

ax1.set_xlim(0.005, 0.9)
ax1.set_ylim(0.0010, 1.1)
ax2.set_xlim(0.005, 0.9)
ax2.set_ylim(0.0010, 1.1)

ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')

fig.tight_layout()

plt.savefig('../Exit/scatter_number_area_log1.png', dpi=300)
plt.close()


###Calculate ratios branchfall to treefall

ratioarea = areabranchfall.area/areatreefall.area
print(ratioarea)
ratioarea.describe().round(3).to_csv('../Exit/ratioarea.csv')


rationumber = numberbranchfall.number/numbertreefall.number
print(rationumber)
rationumber.describe().round(3).to_csv('../Exit/rationumber.csv')