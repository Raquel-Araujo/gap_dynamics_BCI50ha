import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys

def funcdaysarea(table, name, ytick, ax=None):

    #Group by each date and sum areas
    tabdin = pd.pivot_table(table, index='date', values='area_m2', aggfunc=np.sum)
    tabdin = tabdin.reset_index()
    #Rename columns
    gaps = tabdin.set_axis(['date', 'area'], axis=1, inplace=False)

    firstdate = pd.Series('2014-10-02')


    ###Calculate number of interval days

    #Maintain the same
    date1 = gaps.date

    #Add the first date and delete last row (drop=True to start index from 0 again)
    date2 = firstdate.append(gaps.date[:-1]).reset_index(drop=True)

    ##Convert to datetime
    d1 = pd.to_datetime(date1, format='%Y/%m/%d')
    d2 = pd.to_datetime(date2, format='%Y/%m/%d')

    days = (d1 - d2).dt.days.rename('days')

    ###Concatenate 
    gapsdays = pd.concat((gaps, days), axis=1)

    gapsdays.loc[:,'areamonth'] = (gapsdays.area/gapsdays.days)*30
    gapsdays['areapercmonth'] = (gapsdays.areamonth/500000)*100

    #Convert days in integer format to datetime format. Map atributes to each value of the column
    gapsdays['dayscor'] = gapsdays['days'].map(dt.timedelta).astype('timedelta64[D]')

    #Convert date to datetime format
    gapsdays.loc[:,'date1'] = pd.to_datetime(gapsdays.date, format='%Y/%m/%d')


    #Plot
    plt.rc('font', family='Times New Roman', size=12)
    ax = ax or plt.gca()

    ax.axvspan(pd.to_datetime('2014-05-01'), pd.to_datetime('2014-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2015-05-01'), pd.to_datetime('2015-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2016-05-01'), pd.to_datetime('2016-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2017-05-01'), pd.to_datetime('2017-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2018-05-01'), pd.to_datetime('2018-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2019-05-01'), pd.to_datetime('2019-12-31'), alpha=0.5, color='gray')

    ax.bar(gapsdays.date1, gapsdays.areapercmonth, width=-gapsdays.dayscor, align='edge', edgecolor='k', color='lightcoral')

    ax.set_xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))
    ax.set_ylim(bottom = 0)

    ax.xaxis.set_ticklabels([])
    
    if ytick == 1:
        ax.set_ylabel(r'Area (% mo$^{-1}$)', labelpad=15, fontsize=12)
    if ytick == 0:
        ax.yaxis.set_ticklabels([])

    ax.set_title(''+name+'')

    return ax, gapsdays


###############################################################################

def funcdaysnumber(table, name, ytick, ax=None):

    #Group by each date and sum areas
    tabdin = pd.pivot_table(table, index='date', values='area_m2', aggfunc='count')
    tabdin = tabdin.reset_index()
    #Rename columns
    gaps = tabdin.set_axis(['date', 'number'], axis=1, inplace=False)

    firstdate = pd.Series('2014-10-02')


    ###Calculate number of interval days

    #Maintain the same
    date1 = gaps.date

    #Add the first date and delete last row (drop=True to start index from 0 again)
    date2 = firstdate.append(gaps.date[:-1]).reset_index(drop=True)

    ##Convert to datetime
    d1 = pd.to_datetime(date1, format='%Y/%m/%d')
    d2 = pd.to_datetime(date2, format='%Y/%m/%d')


    days = (d1 - d2).dt.days.rename('days')


    ###Concatenate 
    gapsdaysnumber = pd.concat((gaps, days), axis=1)


    gapsdaysnumber.loc[:,'numbermonth'] = (gapsdaysnumber.number/gapsdaysnumber.days)*30
    gapsdaysnumber['numbermonthha'] = (gapsdaysnumber.numbermonth/50)

    #Convert days in integer format to datetime format. Map atributes to each value of the column
    gapsdaysnumber['dayscor'] = gapsdaysnumber['days'].map(dt.timedelta).astype('timedelta64[D]')

    #Convert date to datetime format
    gapsdaysnumber.loc[:,'date1'] = pd.to_datetime(gapsdaysnumber.date, format='%Y/%m/%d')


    #Plot
    plt.rc('font', family='Times New Roman', size=12)
    ax = ax or plt.gca()

    ax.axvspan(pd.to_datetime('2014-05-01'), pd.to_datetime('2014-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2015-05-01'), pd.to_datetime('2015-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2016-05-01'), pd.to_datetime('2016-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2017-05-01'), pd.to_datetime('2017-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2018-05-01'), pd.to_datetime('2018-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2019-05-01'), pd.to_datetime('2019-12-31'), alpha=0.5, color='gray')

    ax.bar(gapsdaysnumber.date1, gapsdaysnumber.numbermonthha, width=-gapsdaysnumber.dayscor, align='edge', edgecolor='k', color='lightcoral')

    ax.set_xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))
    ax.set_ylim(bottom = 0)
    ax.set_xlabel('Images dates', labelpad=10, fontsize=12)

    
    if ytick == 1:
        ax.set_ylabel(r'Number of disturbances (n ha$^{-1}$ mo$^{-1}$)', labelpad=15, fontsize=12)
    if ytick == 0:
        ax.yaxis.set_ticklabels([])


    return ax