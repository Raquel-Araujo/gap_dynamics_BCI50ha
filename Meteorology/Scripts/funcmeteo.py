import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
# from matplotlib.cm import ScalarMappable
# import statsmodels.api as sm
import sys

###Function to produce plots of meteorological data

def functionrainfall (raws, ax=None):
    
    ###Create weekly dates
    ###Example: Round 15 min transforms one 5 min time before and one 5 min after
    raws['datetime7d'] = raws['datetime1'].dt.round('7D')
    # print(raws)

    #Select the columns datetime7d and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain7d = raws.loc[:,['datetime7d','ra']]
    # print(tabrain7d)
    
    #Number of 15 min in 24 hours (24*60)/15
    # constday = 96
    #Number of 15 min in 1 hour = 4
    consthour = 4


    #Group by one week and average the rainfall values
    #Multiply by 96 to have y axis in mm/day. Convert mm/15min to mm/day
    #Multiply by 4 to have y axis in mm/hour. Convert mm/15min to mm/hour


    # rain7d = tabrain7d.groupby((tabrain7d.datetime7d)).sum()
    rain7d = tabrain7d.groupby((tabrain7d.datetime7d)).mean()*consthour

    rain7dr = rain7d.reset_index()
    # print(rain7dr)
    # print(rain7dr.dtypes)

    # plt.plot(rain7dr.datetime7d, rain7dr.ra)
    # plt.show()
    


    ###Create monthly periods
    raws['datetime1m'] = raws['datetime1'].dt.to_period('M')
    # print(raws)

    #Select the columns datetime1m and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain1m = raws.loc[:,['datetime1m','ra']]
    # print(tabrain1m)

    #Group by one month and average the rainfall values
    rain1m = tabrain1m.groupby((tabrain1m.datetime1m)).mean()*consthour
    rain1mr = rain1m.reset_index()

    #Convert monthly periods to datetime
    rain1mr.loc[:,'month'] = rain1mr.datetime1m.dt.strftime('%Y-%m')
    #Sum 14 days to have the mid of each month, as the round used in 7days do the middle
    #I did this to have both lines aligned in the graph.
    rain1mr.loc[:,'month1'] = pd.to_datetime(rain1mr.month, format='%Y-%m')+ dt.timedelta(days=14)

    print(rain7dr)
    print(rain1mr)


    ###Plot rainfall
    # plt.figure(figsize=(11, 5))
    plt.rc('font', family='Times New Roman', size=12)
    ax = ax or plt.gca()

    ax.axvspan(pd.to_datetime('2014-05-01'), pd.to_datetime('2014-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2015-05-01'), pd.to_datetime('2015-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2016-05-01'), pd.to_datetime('2016-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2017-05-01'), pd.to_datetime('2017-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2018-05-01'), pd.to_datetime('2018-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2019-05-01'), pd.to_datetime('2019-12-31'), alpha=0.5, color='gray')

    ax.plot(rain7dr.datetime7d, rain7dr.ra, color='dimgray', label='7-day mean')
    ax.plot(rain1mr.month1, rain1mr.ra, color='k', label='30-day mean')

    # ax.axhspan(meanrain, meanrain, color='k', linewidth=0.5)

    ax.set_xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))

    # ax.set_xlabel('Time (yr)', labelpad=10, fontsize=12)
    ax.set_ylabel(r'Rainfall (mm hour$^{-1}$)', labelpad=15, fontsize=12)
    ax.xaxis.set_ticklabels([])
    ax.legend(loc='upper right')

    # plt.savefig('../Exit/rainfall_sub.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return ax


###############################################################################
###############################################################################
###Function to produce plots of meteorological data

def functionrainfallmonth (raws):

    ###Create weekly dates
    ###Example: Round 15 min transforms one 5 min time before and one 5 min after
    raws['datetime7d'] = raws['datetime1'].dt.round('7D')
    # print(raws)

    #Select the columns datetime7d and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain7d = raws.loc[:,['datetime7d','ra']]
    # print(tabrain7d)
    

    #Group by one week and sum the rainfall values
    rain7d = tabrain7d.groupby((tabrain7d.datetime7d)).sum()

    rain7dr = rain7d.reset_index()
    # print(rain7dr)
    # print(rain7dr.dtypes)

    # plt.plot(rain7dr.datetime7d, rain7dr.ra)
    # plt.show()
    
    ###Create monthly periods
    raws['datetime1m'] = raws['datetime1'].dt.to_period('M')
    # print(raws)

    #Select the columns datetime1m and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain1m = raws.loc[:,['datetime1m','ra']]
    # print(tabrain1m)

    #Group by one month and sum the rainfall values
    rain1m = tabrain1m.groupby((tabrain1m.datetime1m)).sum()
    rain1mr = rain1m.reset_index()

    #Convert monthly periods to datetime
    rain1mr.loc[:,'month'] = rain1mr.datetime1m.dt.strftime('%Y-%m')
    rain1mr.loc[:,'month1'] = pd.to_datetime(rain1mr.month, format='%Y-%m')

    #Export monthly table
    rain1mr.to_csv('../Exit/rain_monthly.csv')

    return rain1mr

###############################################################################
###############################################################################
###Function wind speed

def functionwind (raws, ax=None):

    ###Create weekly dates
    ###Example: Round 15 min transforms one 5 min time before and one 5 min after
    raws['datetime7d'] = raws['datetime1'].dt.round('7D')
    # print(raws)

    #Select the columns datetime7d and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain7d = raws.loc[:,['datetime7d','wsmx']]
    # print(tabrain7d)


    #Group by one week and do the mean of the wind speed values
    rain7d = tabrain7d.groupby((tabrain7d.datetime7d)).mean()
    rain7dr = rain7d.reset_index()
    # print(rain7dr)
    # print(rain7dr.dtypes)

    # plt.plot(rain7dr.datetime7d, rain7dr.ra)
    # plt.show()


    ###Create monthly periods
    raws['datetime1m'] = raws['datetime1'].dt.to_period('M')
    # print(raws)

    #Select the columns datetime1m and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain1m = raws.loc[:,['datetime1m','wsmx']]
    # print(tabrain1m)

    #Group by one month and do the mean of the wind speed values
    rain1m = tabrain1m.groupby((tabrain1m.datetime1m)).mean()
    rain1mr = rain1m.reset_index()

    #Convert monthly periods to datetime
    rain1mr.loc[:,'month'] = rain1mr.datetime1m.dt.strftime('%Y-%m')
    rain1mr.loc[:,'month1'] = pd.to_datetime(rain1mr.month, format='%Y-%m')+ dt.timedelta(days=14)

    meanwind = rain1mr.wsmx.mean()
    print('wind average = ', meanwind)
    # print(rain1mr)
    # print(rain1mr.dtypes)

    ###Plot wind speed
    # plt.figure(figsize=(11, 5))
    plt.rc('font', family='Times New Roman', size=12)
    ax = ax or plt.gca()

    ax.axvspan(pd.to_datetime('2014-05-01'), pd.to_datetime('2014-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2015-05-01'), pd.to_datetime('2015-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2016-05-01'), pd.to_datetime('2016-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2017-05-01'), pd.to_datetime('2017-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2018-05-01'), pd.to_datetime('2018-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2019-05-01'), pd.to_datetime('2019-12-31'), alpha=0.5, color='gray')

    ax.plot(rain7dr.datetime7d, (rain7dr.wsmx)/3.6, color='dimgray', label='7-day mean')
    ax.plot(rain1mr.month1, (rain1mr.wsmx)/3.6, color='k', label='30-day mean')

    # ax.axhspan(meanwind, meanwind, color='k', linewidth=0.5)

    ax.set_xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))

    ax.set_xlabel('Time (yr)', labelpad=10, fontsize=12)
    ax.set_ylabel(r'Wind speed (m s$^{-1}$)', labelpad=15, fontsize=12)
    # ax.legend(loc='upper right')

    # plt.savefig('../Exit/windspeed_sub.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return ax


###############################################################################
###############################################################################
###Function soil humidity

def functionsoilh (sh, ax=None):

    ###Create weekly dates
    sh['date7d'] = sh['date2'].dt.round('7D')
    # print(sh)

    #Select the columns date7d and humidity dry
    tabsoil7d = sh.loc[:,['date7d','h2o_by_dry']]
    # print(tabsoil7d)

    #Group by one week and do the mean of soil humidity values
    soil7d = tabsoil7d.groupby((tabsoil7d.date7d)).mean()
    soil7dr = soil7d.reset_index()
    # print(soil7dr)
    # print(soil7dr.dtypes)
    # plt.plot(soil7dr.date7d, soil7dr.h2o_by_dry, color='k')
    # plt.show()


    #######################################################
    ###Create monthly periods
    sh['date1m'] = sh['date2'].dt.to_period('M')
    # print(sh)

    #Select the columns date1m and soil humidity dry
    #Reason of this: groudby only work with one variable
    tabsoil1m = sh.loc[:,['date1m','h2o_by_dry']]
    # print(tabsoil1m)

    #Group by one month and do the mean of soil humidity values
    soil1m = tabsoil1m.groupby((tabsoil1m.date1m)).mean()
    soil1mr = soil1m.reset_index()

    #Convert monthly periods to datetime
    soil1mr.loc[:,'month'] = soil1mr.date1m.dt.strftime('%Y-%m')
    soil1mr.loc[:,'month1'] = pd.to_datetime(soil1mr.month, format='%Y-%m')

    meansoilh = soil1mr.h2o_by_dry.mean()
    print('soil average = ', meansoilh)

    # print(soil1mr)
    # print(soil1mr.dtypes)
    # plt.plot(soil1mr.month1, soil1mr.h2o_by_dry, color='k')
    # plt.show()

    ###Plot soil humidity
    # plt.figure(figsize=(11, 5))
    plt.rc('font', family='Times New Roman', size=12)
    ax = ax or plt.gca()

    # ax.axvspan(pd.to_datetime('2014-05-01'), pd.to_datetime('2014-11-30'), alpha=0.5, color='gray')
    # ax.axvspan(pd.to_datetime('2015-05-01'), pd.to_datetime('2015-11-30'), alpha=0.5, color='gray')
    # ax.axvspan(pd.to_datetime('2016-05-01'), pd.to_datetime('2016-11-30'), alpha=0.5, color='gray')
    # ax.axvspan(pd.to_datetime('2017-05-01'), pd.to_datetime('2017-11-30'), alpha=0.5, color='gray')
    # ax.axvspan(pd.to_datetime('2018-05-01'), pd.to_datetime('2018-11-30'), alpha=0.5, color='gray')
    # ax.axvspan(pd.to_datetime('2019-05-01'), pd.to_datetime('2019-11-30'), alpha=0.5, color='gray')

    ax.plot(soil7dr.date7d, soil7dr.h2o_by_dry, color='sandybrown', label='Mean Weekly')
    ax.plot(soil1mr.month1, soil1mr.h2o_by_dry, color='sienna', label='Mean Monthly')

    ax.axhspan(meansoilh, meansoilh, color='k', linewidth=0.5)

    ax.set_xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))

    ax.set_xlabel('Time (yr)', labelpad=10, fontsize=12)
    ax.set_ylabel('Soil humidity (%)', labelpad=15, fontsize=12)
    ax.legend(loc='upper right')

    # plt.savefig('../Exit/soilhumidity_sub.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return ax



###############################################################################
###############################################################################
###Function rainfall - with daily maximum


def functionrainfall1 (raws, ax=None):

    ###Create daily dates
    ###Example: Round 15 min transforms one 5 min time before and one 5 min after
    raws['datetime1d'] = raws['datetime1'].dt.round('1D')
    # print(raws)

    #Select the columns datetime1d and ra (rainfall)
    #Reason of this: groudby only work with one variable

    tabrain1d = raws.loc[:,['datetime1d','ra']]
    # print(tabrain1d)
    
    #Number of 15 min in 24 hours (24*60)/15
    # constday = 96
    #Number of 15 min in 1 hour = 4
    consthour = 4


    #Group by one week and average the rainfall values
    #Multiply by 96 to have y axis in mm/day. Convert mm/15min to mm/day
    #Multiply by 4 to have y axis in mm/hour. Convert mm/15min to mm/hour


    # rain1d = tabrain1d.groupby((tabrain1d.datetime1d)).sum()
    rain1d = tabrain1d.groupby((tabrain1d.datetime1d)).max()*consthour

    rain1dr = rain1d.reset_index()
    rain1dr.to_csv('../Exit/rain_max_1day.csv')
    # print(rain7dr)
    # print(rain7dr.dtypes)
    
    
    
    
    ###Create weekly dates
    ###Example: Round 15 min transforms one 5 min time before and one 5 min after
    raws['datetime7d'] = raws['datetime1'].dt.round('7D')
    # print(raws)

    #Select the columns datetime7d and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain7d = raws.loc[:,['datetime7d','ra']]
    # print(tabrain7d)
    
    #Number of 15 min in 24 hours (24*60)/15
    # constday = 96
    #Number of 15 min in 1 hour = 4
    consthour = 4


    #Group by one week and average the rainfall values
    #Multiply by 96 to have y axis in mm/day. Convert mm/15min to mm/day
    #Multiply by 4 to have y axis in mm/hour. Convert mm/15min to mm/hour


    # rain7d = tabrain7d.groupby((tabrain7d.datetime7d)).sum()
    rain7d = tabrain7d.groupby((tabrain7d.datetime7d)).mean()*consthour

    rain7dr = rain7d.reset_index()
    # print(rain7dr)
    # print(rain7dr.dtypes)

    # plt.plot(rain7dr.datetime7d, rain7dr.ra)
    # plt.show()
    


    ###Create monthly periods
    raws['datetime1m'] = raws['datetime1'].dt.to_period('M')
    # print(raws)

    #Select the columns datetime1m and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain1m = raws.loc[:,['datetime1m','ra']]
    # print(tabrain1m)

    #Group by one month and average the rainfall values
    rain1m = tabrain1m.groupby((tabrain1m.datetime1m)).mean()*consthour
    rain1mr = rain1m.reset_index()

    #Convert monthly periods to datetime
    rain1mr.loc[:,'month'] = rain1mr.datetime1m.dt.strftime('%Y-%m')
    #Sum 14 days to have the mid of each month, as the round used in 7days do the middle
    #I did this to have both lines aligned in the graph.
    rain1mr.loc[:,'month1'] = pd.to_datetime(rain1mr.month, format='%Y-%m')+ dt.timedelta(days=14)

    print(rain7dr)
    print(rain1mr)


    ###Plot rainfall
    # plt.figure(figsize=(11, 5))
    plt.rc('font', family='Times New Roman', size=12)
    ax = ax or plt.gca()

    ax.axvspan(pd.to_datetime('2014-05-01'), pd.to_datetime('2014-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2015-05-01'), pd.to_datetime('2015-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2016-05-01'), pd.to_datetime('2016-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2017-05-01'), pd.to_datetime('2017-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2018-05-01'), pd.to_datetime('2018-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2019-05-01'), pd.to_datetime('2019-12-31'), alpha=0.5, color='gray')

    ax.plot(rain1dr.datetime1d, rain1dr.ra, color='dimgray', label='1-day max')
    # ax.plot(rain7dr.datetime7d, rain7dr.ra, color='dimgray', label='7-day mean')
    # ax.plot(rain1mr.month1, rain1mr.ra, color='k', label='30-day mean')

    # ax.axhspan(meanrain, meanrain, color='k', linewidth=0.5)

    ax.set_xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))

    # ax.set_xlabel('Time (yr)', labelpad=10, fontsize=12)
    ax.set_ylabel(r'Rainfall (mm hour$^{-1}$)', labelpad=15, fontsize=12)
    ax.xaxis.set_ticklabels([])
    ax.legend(loc='upper right')

    # plt.savefig('../Exit/rainfall_sub.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return ax



###############################################################################
###############################################################################
###Function wind speed - with daily maximum

def functionwind1 (raws, ax=None):

###Create daily dates
    ###Example: Round 15 min transforms one 5 min time before and one 5 min after
    raws['datetime1d'] = raws['datetime1'].dt.round('1D')
    # print(raws)

    #Select the columns datetime1d and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain1d = raws.loc[:,['datetime1d','wsmx']]
    # print(tabrain1d)


    #Group by one week and do the max of the wind speed values
    rain1d = tabrain1d.groupby((tabrain1d.datetime1d)).max()
    rain1dr = rain1d.reset_index()
    rain1dr.to_csv('../Exit/wind_max_1day.csv')
    # print(rain1dr)
    # print(rain1dr.dtypes)





    ###Create weekly dates
    ###Example: Round 15 min transforms one 5 min time before and one 5 min after
    raws['datetime7d'] = raws['datetime1'].dt.round('7D')
    # print(raws)

    #Select the columns datetime7d and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain7d = raws.loc[:,['datetime7d','wsmx']]
    # print(tabrain7d)


    #Group by one week and do the mean of the wind speed values
    rain7d = tabrain7d.groupby((tabrain7d.datetime7d)).mean()
    rain7dr = rain7d.reset_index()
    # print(rain7dr)
    # print(rain7dr.dtypes)

    # plt.plot(rain7dr.datetime7d, rain7dr.ra)
    # plt.show()


    ###Create monthly periods
    raws['datetime1m'] = raws['datetime1'].dt.to_period('M')
    # print(raws)

    #Select the columns datetime1m and ra (rainfall)
    #Reason of this: groudby only work with one variable
    tabrain1m = raws.loc[:,['datetime1m','wsmx']]
    # print(tabrain1m)

    #Group by one month and do the mean of the wind speed values
    rain1m = tabrain1m.groupby((tabrain1m.datetime1m)).mean()
    rain1mr = rain1m.reset_index()

    #Convert monthly periods to datetime
    rain1mr.loc[:,'month'] = rain1mr.datetime1m.dt.strftime('%Y-%m')
    rain1mr.loc[:,'month1'] = pd.to_datetime(rain1mr.month, format='%Y-%m')+ dt.timedelta(days=14)

    meanwind = rain1mr.wsmx.mean()
    print('wind average = ', meanwind)
    # print(rain1mr)
    # print(rain1mr.dtypes)

    ###Plot wind speed
    # plt.figure(figsize=(11, 5))
    plt.rc('font', family='Times New Roman', size=12)
    ax = ax or plt.gca()

    ax.axvspan(pd.to_datetime('2014-05-01'), pd.to_datetime('2014-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2015-05-01'), pd.to_datetime('2015-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2016-05-01'), pd.to_datetime('2016-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2017-05-01'), pd.to_datetime('2017-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2018-05-01'), pd.to_datetime('2018-12-31'), alpha=0.5, color='gray')
    ax.axvspan(pd.to_datetime('2019-05-01'), pd.to_datetime('2019-12-31'), alpha=0.5, color='gray')

    ax.plot(rain1dr.datetime1d, (rain1dr.wsmx)/3.6, color='dimgray', label='1-day max')
    # ax.plot(rain7dr.datetime7d, (rain7dr.wsmx)/3.6, color='dimgray', label='7-day mean')
    # ax.plot(rain1mr.month1, (rain1mr.wsmx)/3.6, color='k', label='30-day mean')

    # ax.axhspan(meanwind, meanwind, color='k', linewidth=0.5)

    ax.set_xlim(left = pd.to_datetime('2014-10-01'), right = pd.to_datetime('2019-11-30'))

    ax.set_xlabel('Time (yr)', labelpad=10, fontsize=12)
    ax.set_ylabel(r'Wind speed (m s$^{-1}$)', labelpad=15, fontsize=12)
    ax.xaxis.set_ticklabels([])

    # ax.legend(loc='upper right')

    # plt.savefig('../Exit/windspeed_sub.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return ax
