import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

##This code calculates number of days between image intervals
##Creates tables with canopy disturbances rates for area and number
##Calculates basic statistics (table S1)

##Gaps clipped to the plot boundary
gapsall = pd.read_csv('../Entrance/gaps_2014_2019_clip50haplot.csv')

#Group by each date and sum areas
tabdin = pd.pivot_table(gapsall, index='date', values='area_m2', aggfunc=np.sum)
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

gapsdays.round(2).to_csv('../Exit/gaps_area_days_5years.csv')


#################################################################
#################################################################
##Do the same for number of events

#Group by each date and sum areas
tabdin = pd.pivot_table(gapsall, index='date', values='area_m2', aggfunc='count')
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

gapsdaysnumber.round(2).to_csv('../Exit/gaps_number_days_5years.csv')


#################################################################
#################################################################
###Basic stats of temporal variation (area per month)

#Exclude the long interval
gapsdays.drop(14,axis=0,inplace=True)
gapsdaysnumber.drop(14, axis=0, inplace=True)

gapsdays.areamonth.describe().round(1).to_csv('../Exit/describe_area_month.csv')
gapsdaysnumber.numbermonth.describe().round(1).to_csv('../Exit/describe_number_month.csv')
gapsdaysnumber.days.describe().round(1).to_csv('../Exit/describe_days.csv')
gapsdays.area.describe().round(1).to_csv('../Exit/describe_area.csv')
gapsdaysnumber.number.describe().round(1).to_csv('../Exit/describe_number.csv')







