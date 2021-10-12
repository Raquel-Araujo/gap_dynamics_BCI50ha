import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats
import math

import sys

##Gaps with centroids inside the plot
gapsarea=pd.read_csv("../Entrance/gaps_2014_2019_centroid.csv")

##Remove the long interval
# ~ means is not in the list
longint = ['2016-05-18']
gapsarea = gapsarea[~gapsarea['date'].isin(longint)]

stats = gapsarea.area_m2.describe()
stats.round(1).to_csv('../Exit/gaparea_basic_stats.csv')

#Sum = 49495.45 m2
soma = gapsarea.area_m2.sum()
pd.Series(soma).round(1).to_csv('../Exit/total_area.csv')
