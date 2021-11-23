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
### make all figure together
#########################################################################
import matplotlib.image as mpimg 
import matplotlib.gridspec as gridspec

#########################################################################
## reading picture on disk
img = [
mpimg.imread('../Entrance/cumulative_percentage.png'),
mpimg.imread('../Entrance/gap_area_heighdrop_gam.png'),
mpimg.imread('../Entrance/main_gap_size_fits_logc25.png'),
mpimg.imread('../Entrance/stackbar_area_heightdrop.png'),




# mpimg.imread('../Exit/gap_number_Branchfall_bars.png'),
# mpimg.imread('../Exit/gap_number_Treefall_bars.png'),
]
abcd = ["(a)", "(c)" , "(b)" ,"(d)"]
# abcd = ["(a)", "(b)", "(c)"]

fig = plt.figure(figsize=(6.5, 5.4))
#width, height
# fig = plt.figure(figsize=(3, 9))

plt.rc('font', family='Times New Roman', size=10)
#########################################################################
# gridspec inside gridspec
#lines,columns
outer_grid = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0)
#########################################################################
### plot each imagen in subplot
for i in range(4):
    ax = plt.Subplot(fig, outer_grid[i])
    ax.imshow(img[i]   )
    ax.set_xticks([])
    ax.set_yticks([])
    ########
    ### plot leter a b c d 
    ax.text(0, 1, abcd[i], transform=ax.transAxes)
    fig.add_subplot(ax)


# remove all spines
all_axes = fig.get_axes()
for ax in all_axes:
    for sp in ax.spines.values():
        sp.set_visible(False)

# fig.set_tight_layout(True)
#######################
### save in dick
# plt.savefig('../Exit/figure_together.png', dpi=500, bbox_inches='tight')
# fig.set_size_inches(, 7.2)
# plt.xlabel(r'Canopy disturbance area (m$^{2}$)', labelpad=10, loc='center')
plt.savefig('../Exit/figure_together_sizes.png', dpi=800, bbox_inches='tight')

plt.close('all')