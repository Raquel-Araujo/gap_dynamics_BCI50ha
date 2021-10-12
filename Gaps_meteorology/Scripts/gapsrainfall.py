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




###This code relates canopy disturbance area and rainfall


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


######################################################################################
######################################################################################
###Rainfall 15 minutes

#Convert datetime15 to datetime format
meteo.loc[:,'datetime1'] = pd.to_datetime(meteo.datetime15, format='%Y/%m/%d %H:%M:%S')

#Select rainfall until Nov30, 2019 (there is data until Dec 4)
#Last day opened because cut function has opened interval in the last date
rainfall = meteo.loc[meteo.datetime1.between('2014-10-02 00:00:00','2019-11-28 00:00:00'),['datetime1','ra']]

#Add periods
rainfall['per'] = pd.cut(rainfall.datetime1, imgdates, labels=range(1,len(imgdates)),include_lowest=True)

#Rename
rainfall15m = rainfall


######################################################################################
######################################################################################
###Rainfall hour

#Convert datetime15 to datetime format
meteo.loc[:,'datetime1'] = pd.to_datetime(meteo.datetime15, format='%Y/%m/%d %H:%M:%S')

#Select rainfall until Nov30, 2019 (there is data until Dec 4)
#Last day opened because cut function has opened interval in the last date
rainfall = meteo.loc[meteo.datetime1.between('2014-10-02 00:00:00','2019-11-28 00:00:00'),['datetime1','ra']]

#Group by each hour
rainfall['datehour'] = rainfall['datetime1'].dt.to_period("H")

rainfall = pd.pivot_table(data=rainfall, values='ra', index='datehour', aggfunc=np.sum).reset_index()

datehourstr = rainfall.datehour.astype(str)
rainfall['datehour1'] = pd.to_datetime(datehourstr, format='%Y/%m/%d %H:%M:%S')

#Add periods
rainfall['per'] = pd.cut(rainfall.datehour1, imgdates, labels=range(1,len(imgdates)), include_lowest=True)

#Rename
rainfallhour = rainfall



######################################################################################
######################################################################################
###Rainfall day

#Convert datetime15 to datetime format
meteo.loc[:,'datetime1'] = pd.to_datetime(meteo.datetime15, format='%Y/%m/%d %H:%M:%S')

#Select rainfall until Nov30, 2019 (there is data until Dec 4)
#Last day opened because cut function has opened interval in the last date
rainfall = meteo.loc[meteo.datetime1.between('2014-10-02 00:00:00','2019-11-28 00:00:00'),['datetime1','ra']]

#Group by each day
rainfall['date'] = rainfall['datetime1'].dt.date

rainfall = pd.pivot_table(data=rainfall, values='ra', index='date', aggfunc=np.sum).reset_index()
rainfall.date = pd.to_datetime(rainfall.date, format='%Y/%m/%d')

#Add periods
rainfall['per'] = pd.cut(rainfall.date, imgdates, labels=range(1,len(imgdates)), include_lowest=True)
# print(rainfall)

#Rename
rainfallday = rainfall




listrainfall = [rainfall15m, rainfallhour, rainfallday]
listname = ['15m', 'hour', 'day']
listprefix = ['m_p', 'h_p', 'd_p']


i = 0
while i < len(listrainfall):


    ######################################################################################
    ###Input tables
    
    #Filter zero rainfall
    rainfallp = listrainfall[i].loc[listrainfall[i].ra>0]

    #Calculate percentiles 90.0 until 99.9
    #Percentiles calculated on positive rainfall values
    perc = np.percentile(rainfallp.ra, np.arange(90.,100., 0.1))
    num = np.arange(90.,100., 0.1)

    #To not change the entrance name
    entrance = perc

    #I will only need the 1-hour value
    #Value of rainfall of the p98.2
    ###The percentile of 98.2 is 24.3 mm/hour
    pd.Series(np.percentile(rainfallp.ra, 98.2)).to_csv('../Exit/rainfall'+listname[i]+'_equal_p982.csv')


    ##Generate table rainfall greater than or equal 98.2p
    rainhour982 = rainfallp.loc[rainfallp.ra>=np.percentile(rainfallp.ra,98.2)]
    rainhour982.to_csv('../Exit/rainfall'+listname[i]+'_greaterequal_982.csv')



    ######################################################################################
    ###Create input tables and run the metrics

    functions = ['count']


    prefix = map(lambda x: listprefix[i]+'{:.1f}'.format(x),num)

    coletor = []

    j=0
    while j < len(entrance):

        mask = listrainfall[i].ra > entrance[j]
        rainfall_filter = listrainfall[i].loc[mask]
        
        pivotrain = pd.pivot_table(data=rainfall_filter, values='ra', index='per', aggfunc=functions)
        pivotrain.columns = [prefix[j]+'_ranumber']
        coletor.append(pivotrain)
        
        j+=1

    rainmetrics = reduce(lambda df1,df2: df1.join(df2), coletor)


    ######################################################################################
    ###Add area information to the rainfall metrics table - standardized values

    #Adjust index to start from 1
    gaps.index = np.arange(1, len(gaps) + 1)
    rainmetrics_gap = rainmetrics.join(gaps['area'])
    
    rainmetrics_days = rainmetrics.join(gaps[['area','days']])

    #Divide all columns by number of days
    #Exclude the last column days
    #Multiply per 30 to have per metrics per month
    rainstand = (rainmetrics_days.iloc[:,:-1].div(rainmetrics_days.days, axis=0))*30

    #Drop long interval
    rainmetrics_stand = rainstand.drop(15)

    #Rename to join at the end
    rainmetrics_stand.to_csv('../Exit/rain'+listname[i]+'_stand_metrics.csv')

    i+=1



######################################################################################
######################################################################################
###Rainfall analysis - standardized values (per month)

rain15m_stand = pd.read_csv('../Exit/rain15m_stand_metrics.csv').set_index('per')
rainhour_stand = pd.read_csv('../Exit/rainhour_stand_metrics.csv').set_index('per')
rainday_stand = pd.read_csv('../Exit/rainday_stand_metrics.csv').set_index('per')

#Join absolute tables
rainstand = rain15m_stand.iloc[:,0:100].join(rainhour_stand.iloc[:,0:100]).join(rainday_stand).fillna(0)


###Linear fit
metrics = rainstand.columns.values

#Create an empty sumario
sumario = pd.DataFrame(columns=['model', 'a', 'b', 'r2', 'r', 'pvalue', 'n>0'])

#While to do the fit and create the summary results table 
i=0
while i < len(metrics):

    x = np.log(rainstand.loc[:,metrics[i]]+1)
    y = np.log(rainstand.area)

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


# #Choose the highest r values
sumariosort = sumario.sort_values(by='r', ascending=False)
sumario_sel = sumariosort.iloc[1:7,:]
print(sumario_sel)
sumario_sel.to_csv('../Exit/summary_highest_r_rainstand.csv')

print(sumario_sel.index.values)
print(sumario_sel.index.values[0])


##Residual
x = np.log(rainstand.loc[:,sumario_sel.index.values[0]]+1)
y = np.log(rainstand.area)
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


plt.hist(res)
plt.savefig('../Exit/hist_residual.png', dpi=300, bbox_inches='tight')
plt.close()

plt.scatter(ypred,res)
plt.axhline(y=0)
plt.savefig('../Exit/scatter_residual.png', dpi=300, bbox_inches='tight')
plt.close()

#Log scale: p = 0.285 (normal)
stat, p = shapiro(res)
print(stat)
print(p)



#####################################################################################################
#####################################################################################################
#####################################################################################################

#####################################################################################################
#Data for graph of rainfall (mm.hour-1) and percentiles

listrainfall = [rainfall15m, rainfallhour, rainfallday]

#Number of 15 min in 1 hour = 4
const15min = 4
constday = 1./24
# print(constday)

coletor = []

i = 0


while i < len(listrainfall):

    #Filter zero rainfall
    rainfallp = listrainfall[i].loc[listrainfall[i].ra>0]

    #Calculate percentiles 90.0 until 99.9
    #Percentiles calculated on positive rainfall values
    perc = np.percentile(rainfallp.ra, np.arange(90.,100., 0.1))

    if i == 0:
        perc = perc*const15min
    elif i == 2:
        perc = perc*constday

    coletor.append(perc)

    i+=1

print(coletor)

# pd.Series(coletor[1]).to_csv('../Exit/rainfallhour.csv')


num = np.arange(90.,100., 0.1)

#####################################################################################################
#####################################################################################################
#####################################################################################################
####Season data = to color scatter by season

gaps1 = gaps.drop(15)
season = pd.read_csv('../Entrance/seasons.csv')

season1 = season.set_index('per')
# print(season1)

rainstand = rainstand.merge(season1, left_index=True, right_index=True)
gaps1 = gaps1.merge(season1, left_index=True, right_index=True)


#####################################################################################################
#####################################################################################################
#####################################################################################################

#####################################################################################################

###Plot scatter plots, R2 and rainfall rates together (3 subplots)

##Parameters scatter
constha = (1/500000.)*100.
metrics = sumario_sel.index.values


#Best metric
xdry = np.log(rainstand.loc[rainstand.season==0,metrics[0]]+1)
ydry = gaps1.loc[gaps1.season==0, 'areapercmonth']

xwet = np.log(rainstand.loc[rainstand.season==1,metrics[0]]+1)
ywet = gaps1.loc[gaps1.season==1, 'areapercmonth']


a = np.array(sumario_sel.iloc[0, 1])
b = np.array(sumario_sel.iloc[0, 2])
r2 = np.array(sumario_sel.iloc[0, 3])
r = np.array(sumario_sel.iloc[0, 4])


plt.rc('font', family='Times New Roman', size=12)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12, 3))
plt.subplots_adjust(wspace=0.3) #wspace=0.25


ax1.scatter(xdry, ydry,s=60, marker='^', facecolors='none', edgecolors='b', label='Dry season', alpha=0.5) #marker='^',
ax1.scatter(xwet, ywet,s=20, facecolors='royalblue', edgecolors='none', label='Wet season', alpha=0.5)


ax1.set_xlabel(r'Log frequency 15-min periods with rainfall > 98.2$^{th}$ (mo$^{-1}$)', labelpad=10, fontsize=12)
ax1.set_ylabel(r'Canopy disturbance rate (% mo$^{-1}$)', labelpad=10, fontsize=12)
ax1.text(0,1.5,'r = %.2f' % float(r))
ax1.legend(loc='lower right', prop={'size': 10})
ax1.set_yscale('log')
ax1.set_yticks([0.01, 0.05, 0.1, 0.5, 1.5])
ax1.set_yticklabels(['0.01','0.05','0.1','0.5','1.5'])
ax1.set_ylim(0.005, 3)



##Parameters R2 graph

#Positions of metrics
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

    #Entrances
    x = num

    ax2.plot(x,y, color=listcor[i], label=listlabel[i], linewidth=1, zorder=1)

    i += 1

ax2.scatter(98.2, 0.46,s=20, edgecolors='r', facecolors='none', zorder=2)
ax2.legend(loc='upper left', prop={'size': 10})
ax2.set_ylabel('Pearson correlation (r)', labelpad=10)
ax2.set_xlabel('Rainfall percentile thresholds', labelpad=10)


ax1.set_title('(a)', loc='left')
ax2.set_title('(b)', loc='left')
ax3.set_title('(c)', loc='left')


ax2.set_xticks(np.arange(90.,100., 1))
ax2.set_xticklabels(namesxticks, rotation=45)

ax2.set_ylim(-0.1, 0.9)

#Proportion of ylimites, scale transformation

ymax3 = (21.84/90)-0.007
xmax3 = (98.2/99.9)-0.185

ax3.plot(num, coletor[0], color='darkmagenta', linewidth=1, label='Rainfall 15-min')
ax3.plot(num, coletor[1], color='royalblue', linewidth=1, label='Rainfall 1-hour')
ax3.plot(num, coletor[2], color='k', linewidth=1, label='Rainfall 1-day')


ax3.set_ylabel('Rainfall rate (mm.hour$^{-1}$)')
ax3.set_xlabel('Rainfall percentile thresholds', labelpad=10)
ax3.set_xticks([90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
ax3.set_xticklabels(namesxticks, rotation=45)

ax3.minorticks_on()
ax3.legend(loc='upper left', prop={'size': 10})

ax3.axvline(x=98.2, ymin=0.0001, ymax=ymax3,ls='--', color='r', linewidth=0.5, zorder=2)
ax3.axhline(y=24.3, xmin=0.0001, xmax=xmax3,ls='--', color='r', linewidth=0.5, zorder=2)

plt.savefig('../Exit/graphs_scatter_r_rainfall_stand2.png', dpi=300, bbox_inches='tight')
plt.close()