import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.cm import ScalarMappable
import collections
import statsmodels.api as sm
import sys
import statsmodels.formula.api as smf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy import stats
from scipy.stats import shapiro




###This code is for relations of gap area and wind speed


gaps = pd.read_csv('../Entrance/gaps_area_days_5years.csv')
meteo = pd.read_csv('../Entrance/tabmeteorology.csv')

######################################################################################
######################################################################################
###Gaps

#Convert date to datetime format
gaps.loc[:,'date1'] = pd.to_datetime(gaps.date, format='%Y/%m/%d')

#Convert days in integer format to datetime format. Map atributes to each value of the column
gaps['dayscor'] = gaps['days'].map(dt.timedelta)

###Vector of image dates
dates = gaps.date
dates.index = dates.index + 1
datestart = pd.Series('2014-10-02')
datesall = datestart.append(dates)

#Convert dateall to datetime format
imgdates = pd.to_datetime(datesall, format='%Y/%m/%d')
print(imgdates)


######################################################################################
######################################################################################
###wind 15 minutes

#Convert datetime15 to datetime format
meteo.loc[:,'datetime1'] = pd.to_datetime(meteo.datetime15, format='%Y/%m/%d %H:%M:%S')

#Select wind until Nov30, 2019 (there is data until Dec 4)
#Last day opened because cut function has opened interval in the last date
wind = meteo.loc[meteo.datetime1.between('2014-10-02 00:00:00','2019-11-28 00:00:00'),['datetime1','wsmx']]

#Add periods
wind['per'] = pd.cut(wind.datetime1, imgdates, labels=range(1,len(imgdates)),include_lowest=True)
print(wind)

#Rename
wind15m = wind


######################################################################################
######################################################################################
###wind hour

#Convert datetime15 to datetime format
meteo.loc[:,'datetime1'] = pd.to_datetime(meteo.datetime15, format='%Y/%m/%d %H:%M:%S')

#Select wind until Nov30, 2019 (there is data until Dec 4)
#Last day opened because cut function has opened interval in the last date
wind = meteo.loc[meteo.datetime1.between('2014-10-02 00:00:00','2019-11-28 00:00:00'),['datetime1','wsmx']]
print(wind)

#Group by each hour
wind['datehour'] = wind['datetime1'].dt.to_period("H")

wind = pd.pivot_table(data=wind, values='wsmx', index='datehour', aggfunc=np.max).reset_index()
print(wind)


datehourstr = wind.datehour.astype(str)
wind['datehour1'] = pd.to_datetime(datehourstr, format='%Y/%m/%d %H:%M:%S')
# print(wind)

#Add periods
wind['per'] = pd.cut(wind.datehour1, imgdates, labels=range(1,len(imgdates)), include_lowest=True)
print(wind)

#Rename
windhour = wind



######################################################################################
######################################################################################
###wind day

#Convert datetime15 to datetime format
meteo.loc[:,'datetime1'] = pd.to_datetime(meteo.datetime15, format='%Y/%m/%d %H:%M:%S')

#Select wind until Nov30, 2019 (there is data until Dec 4)
#Last day opened because cut function has opened interval in the last date
wind = meteo.loc[meteo.datetime1.between('2014-10-02 00:00:00','2019-11-28 00:00:00'),['datetime1','wsmx']]

#Group by each day
wind['date'] = wind['datetime1'].dt.date

wind = pd.pivot_table(data=wind, values='wsmx', index='date', aggfunc=np.max).reset_index()
wind.date = pd.to_datetime(wind.date, format='%Y/%m/%d')

#Add periods
wind['per'] = pd.cut(wind.date, imgdates, labels=range(1,len(imgdates)), include_lowest=True)
# print(wind)

#Rename
windday = wind





listwind = [wind15m, windhour, windday]
listname = ['15m', 'hour', 'day']
listprefix = ['m_p', 'h_p', 'd_p']


i = 0
while i < len(listwind):


    ######################################################################################
    ###Input tables
    
    #Filter zero wind
    windp = listwind[i].loc[listwind[i].wsmx>0]
    # print(windp)

    #Calculate percentiles 90.0 until 99.9
    #Percentiles calculated on positive wind values
    perc = np.percentile(windp.wsmx, np.arange(90.,100., 0.1))
    num = np.arange(90.,100., 0.1)
    # print(num)
    # print(perc)
    # print(len(num))
    # print(len(perc))


    #To not change the entrance name
    entrance = perc
    # print(entrance)


    ######################################################################################
    ###Create input tables and run the metrics

    functions = ['count']


    prefix = map(lambda x: listprefix[i]+'{:.1f}'.format(x),num)
    # prefix = ['all', 'p']+pref
    # print(prefix)

    # print(len(prefix))
    # print(len(entrance))

    coletor = []

    j=0
    while j < len(entrance):

        mask = listwind[i].wsmx > entrance[j]
        wind_filter = listwind[i].loc[mask]
        
        pivotwind = pd.pivot_table(data=wind_filter, values='wsmx', index='per', aggfunc=functions)
        pivotwind.columns = [prefix[j]+'_wsnumber']
        coletor.append(pivotwind)
        
        j+=1

    windmetrics = reduce(lambda df1,df2: df1.join(df2), coletor)
    # print(windmetrics)



    ######################################################################################
    ###Add area information to the wind metrics table - standardized values

    #Adjust index to start from 1
    gaps.index = np.arange(1, len(gaps) + 1)
    windmetrics_gap = windmetrics.join(gaps['area'])
    

    windmetrics_days = windmetrics.join(gaps[['area','days']])
    # print(windmetrics_days)

    #Divide all columns by number of days
    #Exclude the last column days
    #Multiply per 30 to have per metrics per month
    windstand = (windmetrics_days.iloc[:,:-1].div(windmetrics_days.days, axis=0))*30
    # print(windstand)

    #Drop long interval
    windmetrics_stand = windstand.drop(15)
    # print(windmetrics_stand)


    #Rename to join at the end
    windmetrics_stand.to_csv('../Exit_ws/wind'+listname[i]+'_stand_metrics.csv')

    i+=1


# print(windmetrics_stand)


######################################################################################
######################################################################################
###wind analysis - standardized values (per month)

wind15m_stand = pd.read_csv('../Exit_ws/wind15m_stand_metrics.csv').set_index('per')
windhour_stand = pd.read_csv('../Exit_ws/windhour_stand_metrics.csv').set_index('per')
windday_stand = pd.read_csv('../Exit_ws/windday_stand_metrics.csv').set_index('per')

# print(windday_stand)

#Join absolute tables
windstand = wind15m_stand.iloc[:,0:100].join(windhour_stand.iloc[:,0:100]).join(windday_stand).fillna(0)
print(windstand)


###Linear fit
metrics = windstand.columns.values
# print(metrics)

#Create an empty sumario
sumario = pd.DataFrame(columns=['model', 'a', 'b', 'r2', 'r', 'pvalue', 'n>0'])
# print(sumario)

#While to do the fit and create the summary results table 
i=0
while i < len(metrics):

    x = np.log(windstand.loc[:,metrics[i]]+1)
    y = np.log(windstand.area)

    x1 = sm.add_constant(x)

    model = sm.OLS(y, x1)
    results = model.fit()

    name = metrics[i]
    modell = 'a+bx'
    a = results.params[0]
    b = results.params[1]
    r2 = results.rsquared
    r = stats.pearsonr(x, y)
    p = results.pvalues[1]
    n = len(x[x>0])

    sumario.loc[metrics[i],:] = modell, a, b, r2, r[0], p, n

    i += 1


#Choose the highest r values
sumariosort = sumario.sort_values(by='r', ascending=False)
print(sumariosort)

sumario_sel = sumariosort.iloc[1:7,:]
sumario_sel.to_csv('../Exit_ws/summary_highest_r_windstand.csv')



##Residual
x = np.log(windstand.loc[:,sumario_sel.index.values[0]]+1)
y = np.log(windstand.area)
x1 = sm.add_constant(x)

model = sm.OLS(y, x1)
results = model.fit()

modell = 'a+bx'
a = results.params[0]
b = results.params[1]
r2 = results.rsquared
r = stats.pearsonr(x, y)
p = results.pvalues[1]
n = len(x[x>0])
# ypred = a + b*x
ypred = results.predict(x1)
# res = y - ypred
res = results.resid
print(a)


plt.hist(res)
plt.savefig('../Exit_ws/hist_residual.png', dpi=300, bbox_inches='tight')
plt.close()

plt.scatter(ypred,res)
plt.axhline(y=0)
plt.savefig('../Exit_ws/scatter_residual.png', dpi=300, bbox_inches='tight')
plt.close()

#Log scale: p = 0.100399978459 (normal)
stat, p = shapiro(res)
print(stat)
print(p)



#####################################################################################################
#####################################################################################################
#####################################################################################################

#####################################################################################################
#Data for graph of windspeed (mm.s-1) and percentiles

listwind = [wind15m, windhour, windday]
listwindname = ['wind15m', 'windhour', 'windday']

# #Number of 15 min in 1 hour = 4
# const15min = 4
# constday = 1./24
# print(constday)

coletor = []

i = 0


while i < len(listwind):

    #Filter zero wind
    windp = listwind[i].loc[listwind[i].wsmx>0]

    #I will only need the 1-day value
    #Value of windspeed of the p99.3
    ###The percentile of 99.3 is 11 m/s
    pd.Series(np.percentile(windp.wsmx, 99.3)/3.6).to_csv('../Exit_ws/wind_ms'+listwindname[i]+'_equal_p993.csv')

    #Calculate percentiles 90.0 until 99.9
    #Percentiles calculated on positive wind values
    perc = np.percentile(windp.wsmx, np.arange(90.,100., 0.1))

    # if i == 0:
    #     perc = perc*const15min
    # elif i == 2:
    #     perc = perc*constday

    coletor.append(perc)

    i+=1

print(coletor)

num = np.arange(90.,100., 0.1)
print(num)

#####################################################################################################
#####################################################################################################
#####################################################################################################
####Season data = to color scatter by season

# print(windstand)
# print(gaps1)
gaps1 = gaps.drop(15)
season = pd.read_csv('../Entrance/seasons.csv') #the file is in exit folder

season1 = season.set_index('per')
# print(season1)

windstand = windstand.merge(season1, left_index=True, right_index=True)
gaps1 = gaps1.merge(season1, left_index=True, right_index=True)

# print(gaps1)


#####################################################################################################
#####################################################################################################
#####################################################################################################

#####################################################################################################

###Plot scatter plots, R2 and wind rates together (3 subplots)

##Parameters scatter
constha = (1/500000.)*100.
metrics = sumario_sel.index.values

# x = windstand.loc[:,metrics[0]]
# y = gaps1.areapercmonth

xdry = np.log(windstand.loc[windstand.season==0,metrics[0]]+1)
ydry = gaps1.loc[gaps1.season==0, 'areapercmonth']

xwet = np.log(windstand.loc[windstand.season==1,metrics[0]]+1)
ywet = gaps1.loc[gaps1.season==1, 'areapercmonth']

# print(xdry)
# print(ydry)

a = np.array(sumario_sel.iloc[0, 1])
b = np.array(sumario_sel.iloc[0, 2])
r2 = np.array(sumario_sel.iloc[0, 3])
r = np.array(sumario_sel.iloc[0, 4])
p = np.array(sumario_sel.iloc[0, 5])

xplot = np.array([np.min(x), np.max(x)])
yplot = (a+b*xplot)*constha
print(yplot)
print(xplot)
print(a)
print(b)



#Multiply the coeficients by constha to write the equation on plot
#I checked this in excel

aha = a*constha
bha = b*constha
print(aha)
print(bha)


plt.rc('font', family='Times New Roman', size=12)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12, 3))
plt.subplots_adjust(wspace=0.3) #wspace=0.25

# ax1.scatter(xwet, ywet,s=30, facecolors='none', edgecolors='royalblue', label='Wet season')
# ax1.scatter(xdry, ydry,s=30, marker='^', facecolors='m', edgecolors='none', label='Dry season', alpha=0.5)


ax1.scatter(xdry, ydry,s=60, marker='^', facecolors='none', edgecolors='k', label='Dry season', alpha=0.5) #marker='^',
ax1.scatter(xwet, ywet,s=20, facecolors='dimgrey', edgecolors='none', label='Wet season', alpha=0.5)


# ax1.scatter(xwet, ywet,s=30, facecolors='royalblue', edgecolors='none', label='Wet season', alpha=0.4)

# ax1.plot(xplot,yplot, color='k', linestyle='--', linewidth=1)

ax1.set_xlabel(r'Log frequency periods with 1-day max windspeed > 99.3$^{th}$ (mo$^{-1}$)', labelpad=10, fontsize=12)
ax1.set_ylabel(r'Canopy disturbance rate (% mo$^{-1}$)', labelpad=10, fontsize=12)
ax1.text(0.65,0.45,'p-value = %.2f' % float(p))
ax1.legend(loc='best', prop={'size': 10})
# ax1.text(0,1.3,'y = %.2f + %.2fx' % (float(aha),float(bha)))
ax1.set_yscale('log')
ax1.set_yticks([0.01, 0.05, 0.1, 0.5, 1.5])
ax1.set_yticklabels(['0.01','0.05','0.1','0.5','1.5'])
# ax1.set_xlim(-0.1, 2.1)
ax1.set_ylim(0.005, 3)



##Parameters R2 graph

#100 is the number of percentiles
listcount15m = np.arange(0,100,1)
listcounthour = 100+listcount15m
listcountday = 100+100+listcount15m
listall = [listcount15m, listcounthour, listcountday]

listcor = ['darkmagenta', 'royalblue', 'k']
listlabel = ['Frequency 15-min', 'Frequency 1-hour', 'Frequency 1-day']

namesxticks = ['90', '91', '92', '93', '94', '95', '96', '97', '98', '99']


num = np.arange(90.,100., 0.1)

#Position of ticks
xtickspos = np.arange(0,100,10)


i = 0
while i < len (listall):

    #r values of metrics
    y = np.array(sumario.iloc[listall[i],4])
    # print(y)

    #Entrances
    # x = np.arange(1,len(y)+1, 1)
    x = num
    # print(x)

    # ax2.scatter(x,y, color=listcor[i], s=2)
    ax2.plot(x,y, color=listcor[i], label=listlabel[i], linewidth=1, zorder=1)

    i += 1

ax2.scatter(99.3, 0.21,s=20, edgecolors='r', facecolors='none', zorder=2)
ax2.legend(loc='upper left', prop={'size': 10})
ax2.set_ylabel('r', labelpad=10)
ax2.set_xlabel('Wind speed percentile thresholds', labelpad=10)
# ax2.xaxis.set_tick_params(axis=0, which='minor', bottom=True )

# ax2.set_xticks(xtickspos,minor=True)
# ax2.set_xticks(np.arange(90.,100., 0.1),minor=True)
# ax2.set_xlim([95.5,99.9])

ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')
ax3.set_title('(c)', loc='left')


ax2.set_xticks(np.arange(90.,100., 1))
ax2.set_xticklabels(namesxticks, rotation=45)

# ax2.set_ylim(-0.25, 0.25)

#Proportion of ylimites, scale transformation
# ymax = (0.67-(-0.1))/(0.9-(-0.1))
# ax2.axvline(x=99.4, ymin=-0.1, ymax=ymax, ls='--', color='gray', linewidth=0.5, zorder=1)

# ymax3 = (11/12)-0.2
# ymax3 = 0.7

xmax3 = (99.3/99.9)-0.0875

ax3.plot(num, coletor[0]/3.6, color='darkmagenta', linewidth=1, label='Wind 15-min')
ax3.plot(num, coletor[1]/3.6, color='royalblue', linewidth=1, label='Wind 1-hour')
ax3.plot(num, coletor[2]/3.6, color='k', linewidth=1, label='Wind 1-day')


ax3.set_ylabel('Wind speed (m s$^{-1}$)')
ax3.set_xlabel('Wind speed percentile thresholds', labelpad=10)
ax3.set_xticks([90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
ax3.set_xticklabels(namesxticks, rotation=45)

ax3.minorticks_on()
ax3.legend(loc='upper left', prop={'size': 10})

ax3.axvline(x=99.3, ymin=0, ymax=0.785,ls='--', color='r', linewidth=0.5, zorder=2)
ax3.axhline(y=11.01, xmin=0.0001, xmax=xmax3,ls='--', color='r', linewidth=0.5, zorder=2)


# ax2.set_xticks(xtickspos)
# ax2.set_xticklabels(namesxticks)
plt.savefig('../Exit_ws/graphs_scatter_r_wind_stand2.png', dpi=300, bbox_inches='tight')
plt.close()