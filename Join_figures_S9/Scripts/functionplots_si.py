import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
# import function as f
# import function1 as f1
# import funcfit as ffit
import sys
import math


#################################################################################################################################################################################
### make all figures together
#########################################################################
import matplotlib.image as mpimg 
import matplotlib.gridspec as gridspec

#########################################################################
## reading picture on disk
img = [
mpimg.imread('../Entrance/gap_size_fits_Exp.png'),
mpimg.imread('../Entrance/gap_size_fits_Power.png'),
mpimg.imread('../Entrance/gap_size_fits_Weibull.png')
]
# abcd = ["a", "b" , "c" ,"d"]
abcd = ["(a)", "(b)", "(c)"]

# fig = plt.figure(figsize=(6.5, 5.4))
#width, height
fig = plt.figure(figsize=(10, 3))

plt.rc('font', family='Times New Roman', size=10)
#########################################################################
# gridspec inside gridspec
#lines,columns
outer_grid = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0)
#########################################################################
### plot each image in subplot
for i in range(3):
    ax = plt.Subplot(fig, outer_grid[i])
    ax.imshow(img[i]   )
    ax.set_xticks([])
    ax.set_yticks([])
    ########
    ### plot letters a b c d 
    ax.text(0, 1, abcd[i], transform=ax.transAxes)
    fig.add_subplot(ax)


# remove all spines
all_axes = fig.get_axes()
for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)


plt.savefig('../Exit/figure_together_si_fits.png', dpi=800, bbox_inches='tight')

plt.close('all')