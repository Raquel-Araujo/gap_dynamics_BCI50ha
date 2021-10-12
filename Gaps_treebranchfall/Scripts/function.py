import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys

def funcdaysarea(table, name):

    #Group by each date and sum areas
    tabdin = pd.pivot_table(table, index='date', values='area_m2', aggfunc=np.sum)
    tabdin = tabdin.reset_index()
    #Rename columns
    gaps = tabdin.set_axis(['date', 'area'], axis=1, inplace=False)
    print(gaps)

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


    return(gapsdays)


###############################################################################

def funcdaysnumber(table, name):

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


    return(gapsdaysnumber)