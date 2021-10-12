import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
# import function as f
import sys, math
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro


###Seasonal patterns (Figure 4)

gaps = pd.read_csv('../Entrance/gaps_area_days_5years.csv')
gaps.loc[:,'date1'] = pd.to_datetime(gaps.date, format='%Y/%m/%d')

gaps['month'] = gaps['date1'].dt.month
gaps['day'] = gaps['date1'].dt.day


#Multiply by 1 transforms True/False into 0 and 1
# 1 = wet (May to December)
# 0 = dry (January to April)
gaps['season'] = (gaps.month>=5)*1
gaps.to_csv('../Exit/gaps_season.csv')


#List of intervals - transitions between seasons
#Remove just the long interval to increase the number of observations

# transitions = ['2015-01-10', '2015-05-20', '2016-05-18', '2017-03-15', '2017-05-17', '2018-02-28', '2018-05-23', '2019-01-28', '2019-05-31']
transitions = ['2016-05-18']

#Gaps full season - remove the long interval
# ~ means is not in the list
gapsfseason = gaps[~gaps['date'].isin(transitions)]


##Change the seasons of 20150110 and 20180523 (more days in December and April, respectively)
gapsfseason.loc[2,'season']=1
gapsfseason.loc[29,'season']=0
print(gapsfseason)

gapsfseason.to_csv('../Exit/gaps_full_season.csv')

#Separate into dry and wet seasons
gapsdry = gapsfseason.loc[gapsfseason['season'] == 0]
gapswet = gapsfseason.loc[gapsfseason['season'] == 1]

##Separate into early and late wet season
#Early = 0
#Late =1
gapswet['earlylate'] = (gapswet.month>=9)*1
#More days in December for the 20150110
gapswet.loc[2,'earlylate']=1
print(gapswet)


#########################
#Export describe 

##Dry and wet season
describewet = gapswet.areapercmonth.describe()
describewet.to_csv('../Exit/wetseason_areapercmonth.csv')
describedry = gapsdry.areapercmonth.describe()
describedry.to_csv('../Exit/dryseason_areapercmonth.csv')

##Early and late wet season
earlywet = gapswet.loc[gapswet['earlylate'] == 0]
latewet = gapswet.loc[gapswet['earlylate'] == 1]
describeearlywet = earlywet.areapercmonth.describe()
describeearlywet.to_csv('../Exit/earlywet_areapercmonth.csv')
describelatewet = latewet.areapercmonth.describe()
describelatewet.to_csv('../Exit/latewet_areapercmonth.csv')


#########################
###Test variances and means
#Use log because the distribution is not normal

###Wet and dry season
areawet = np.array(np.log(gapswet.areapercmonth))
areadry = np.array(np.log(gapsdry.areapercmonth))

###Normality test
x = np.concatenate((areawet, areadry))
stat, p = shapiro(x)

#p value  = 0.065
#p > 0.05: 'Sample looks Gaussian (fail to reject H0)'

pnormalwetdry = pd.Series(p)
pnormalwetdry.to_csv('../Exit/pvalues_wetdry_normalshapiro.csv')


#Levene
statistic, pvalue_lev = stats.levene(areawet, areadry, center='median')
###p=0.5214 > 0.05, this means accept the null hypothesis of equal variaces.
##Variances are equal for wet and dry seasons

#T-test
statistic, pvalue_t = stats.ttest_ind(areawet, areadry)

pwetdry = pd.DataFrame([pvalue_lev, pvalue_t], index=['pvalue_lev', 'pvalue_t'])
pwetdry.to_csv('../Exit/pvalues_wetdry.csv')



#############################################
###Early and late wet season
areaearlywet = np.array(np.log(earlywet.areapercmonth))
arealatewet = np.array(np.log(latewet.areapercmonth))


##Normality test
xwet = np.concatenate((areaearlywet, arealatewet))
stat, p = shapiro(xwet)

#p value  = 0.09
#p > 0.05: 'Sample looks Gaussian (fail to reject H0)'

pnormalearlylate_wet = pd.Series(p)
pnormalearlylate_wet.to_csv('../Exit/pvalues_earlylate_wet_normalshapiro.csv')


statistic, pvalue_lev = stats.levene(areaearlywet, arealatewet, center='median')
 
###p=0.4872 < 0.05
##Variances are not equal for early and late wet seasons

# The two-tailed p-value
statistic, pvalue_t = stats.ttest_ind(areaearlywet, arealatewet, equal_var=False)

pearlylatewet = pd.DataFrame([pvalue_lev, pvalue_t], index=['pvalue_lev', 'pvalue_t'])
pearlylatewet.to_csv('../Exit/pvalues_earlylate_wet.csv')


###Violin plot scaled by number of observations
##Values in natural log (base e) 
labels = ['Dry', 'Wet']
labelswet = ['Early Wet', 'Late Wet']

plt.rc('font', family='Times New Roman', size=12)
fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8,4))

#Confidence intervals 95%
sns.violinplot(ax = ax1, x = gapsfseason['season'], y = np.log(gapsfseason['areapercmonth']), palette=['indianred','steelblue'], inner=None, cut=0, scale='count')
sns.pointplot(ax = ax1, x = gapsfseason['season'], y = np.log(gapsfseason['areapercmonth']), estimator=np.mean, color='black')

sns.violinplot(ax = ax2, x = gapswet['earlylate'], y = np.log(gapswet['areapercmonth']), palette="Blues", inner=None, cut=0, scale='count')
sns.pointplot(ax = ax2, x = gapswet['earlylate'], y = np.log(gapswet['areapercmonth']), estimator=np.mean, color='black')

ax1.set_ylim(top=1)
ax2.set_ylim(top=1)

ax1.set_xticks(np.arange(0, len(labels)))
ax1.set_xticklabels(labels)

ticklabel = [0.01, 0.05, 0.15, 0.5, 1, 1.5]

tick = np.log(ticklabel)
ax1.set_yticks(tick)
ax1.set_yticklabels(ticklabel)

ax2.set_xticks(np.arange(0, len(labelswet)))
ax2.set_xticklabels(labelswet)

ax2.set_yticks(tick)
ax2.set_yticklabels(ticklabel)

ax1.set_xlabel('')
ax2.set_xlabel('')

ax1.set_ylabel(r'Canopy disturbance rate (% mo$^{-1}$)', labelpad=15)
ax2.set_ylabel('')

ax1.text(1, -4.6, 'p = %.2f' % pwetdry.loc['pvalue_t',0])
ax2.text(1, -4.35, 'p = %.2f' % pearlylatewet.loc['pvalue_t',0])

ax1.text(0, -1.1, int(describedry['count']), horizontalalignment='center')
ax1.text(1, 0.6, int(describewet['count']), horizontalalignment='center')
ax2.text(0, 0.6, int(describeearlywet['count']), horizontalalignment='center')
ax2.text(1, 0.1, int(describelatewet['count']), horizontalalignment='center')

ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')

plt.subplots_adjust(hspace = 0.2)
plt.savefig('../Exit/violinplot_scale_log1.png', dpi=300, bbox_inches='tight')
plt.close()
