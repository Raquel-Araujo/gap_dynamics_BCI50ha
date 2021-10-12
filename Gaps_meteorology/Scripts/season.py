import numpy as np
import pandas as pd

tabseason = pd.read_csv('../Entrance/gaps_full_season.csv')
print(tabseason)
print(tabseason.columns)

season = tabseason.loc[:,['Unnamed: 0', 'season']]
season.columns = ['per', 'season']
season['per'] = season.per + 1

season1 = season.set_index('per')
print(season1)

season1.to_csv('../Entrance/seasons.csv')

