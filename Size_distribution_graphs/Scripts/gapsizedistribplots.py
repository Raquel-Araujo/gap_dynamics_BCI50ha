import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functionsmle1 as f
# import funcbootstrap as fb
# from scipy import stats
# import funcfit as ffit
# import math
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


##Gaps with centroids inside the plot
gapsizeclass=pd.read_csv("../Entrance/gaps_2014_2019_centroid.csv")

#Fits from continuous analysis R code
tabresultall1 = pd.read_csv('../Entrance/allsizedistfitsnoci.csv')
#Selecting maximum size 50 ha
tabresultall = tabresultall1.iloc[12:24,:]


########################################################################################################
########################################################################################################

########################################################################################################
###Plot with different size classes

#Classes
listclasse = [
[2,5,10,15,20,25,30,35,40,45,50,55,60,70,80,90], #intervals 3,5,15
np.arange(100, 121, 20),
np.arange(150, 451, 50),
] 


#Function to concatenate
dbhmin = reduce(lambda x,y: np.concatenate((x,y)), listclasse)
bins = np.concatenate((dbhmin, [500]))
dbhmax = np.concatenate((dbhmin[1:],[500])) ###remove the first and add 500

width = dbhmax - dbhmin
label = dbhmin+(width/2.)

gapsizeclass.columns = ['id', 'area', 'date']
gapsizeclass.loc[:,('classe')] = pd.cut(gapsizeclass['area'], bins, labels=dbhmin)


#Pivot table
freqclass = pd.pivot_table(gapsizeclass, index="classe", values="area", aggfunc=len).reset_index()
freqclass.columns = ['classe', 'n']
freqclass.to_csv('../Exit/frequency_classes.csv')
freqclass["classe"] = freqclass["classe"].astype('int')



i=0

valores = [2, 5, 10, 25]
nomes = ['c2', 'c5', 'c10', 'c25']
nomestitle = [' > 2m$^{2}$', ' > 5m$^{2}$', ' > 10m$^{2}$', ' > 25m$^{2}$']
valoresylinha = [0.69, 0.85, 0.85, 0.87]


while i < len(valores):

    frequenciac = freqclass.loc[freqclass['classe']>=valores[i]]

    #This is boolean result
    mask = freqclass['classe']>=valores[i]

    #Select only the true values
    width1 = width[mask]


    ##Plot
    size = tabresultall.loc[tabresultall['minsize'] == valores[i],:]

    parexp = size.iloc[0,4]
    parpower = size.iloc[1,4]
    parweibull = size.iloc[2,[5,4]]
    # Weibull parameters order is different in the original code

    dbhmin = frequenciac.classe
    dbhmax = dbhmin+width1

    mindbh = dbhmin.min()
    maxdbh = dbhmax.max()

    ##Values per hectare (total of 50 hectares)
    constha = 0.02
    div = np.sum(frequenciac.n)*constha/width1

    #Plot loglog
    plt.figure(figsize=(4, 3))
    plt.rc('font', family='Times New Roman', size=12)
    plt.scatter(freqclass.classe+(width/2.), (freqclass.n)*constha/width, facecolors='none', edgecolors='k',s=20, linewidths=1, label='_nolegend_')

    plt.plot(frequenciac.classe+(width1/2.), f.funcexp(dbhmin, dbhmax, parexp, mindbh, maxdbh)*div, color='dimgrey', linewidth=0.8, label='Exp')
    plt.plot(frequenciac.classe+(width1/2.), f.funcpower(dbhmin, dbhmax, parpower, mindbh, maxdbh)*div, color='blue', linewidth=0.8, label='Power')
    plt.plot(frequenciac.classe+(width1/2.), f.funcweibull(dbhmin, dbhmax, parweibull[0],parweibull[1], mindbh, maxdbh)*div, color='red', linewidth=1.2, linestyle='--', label='Weibull', dashes=(5, 5))

    plt.loglog()
    plt.ylim(0.00001,3)
    plt.xlim(1,600)
    plt.legend(loc='best')
    plt.title('Area'+nomestitle[i])
    plt.xlabel(r'Canopy disturbance area (m$^{2}$)', labelpad=10)
    plt.ylabel(r'Frequency (n.ha$^{-1}$.m$^{2}$)', labelpad=10)

    #Proportion y
    plt.axvline(x=valores[i], ymin=0.0001, ls='--', color='k', linewidth=0.5, zorder=1) #ymax=valoresylinha[i]


    plt.savefig('../Exit/gap_size_fits_log'+nomes[i]+'.png', dpi=300, bbox_inches='tight')

    plt.close()

    i+=1



########################################################################################################
########################################################################################################
########################################################################################################
##Plot SI2

# listacolor = ['Greys' ,'Blues', 'Reds']
listacolor = ['Set1', 'Set1', 'Set1']

colorcoletor = []
for c in listacolor:

    cmap = plt.cm.get_cmap(c)
    # rgba = cmap([0.5, 0.6, 0.7, 0.8])
    rgba = cmap([0.1, 0.2, 0.4, 0.5])


    colorcoletor.append(rgba)


listafunc = [f.funcexp, f.funcpower, f.funcweibullmod]
listanome = ['Exp', 'Power', 'Weibull']
listalinestyle = ['-', '-', '--']


j = 0
while j < len(listafunc):

    #Plot loglog
    plt.figure(figsize=(4, 3))
    plt.rc('font', family='Times New Roman', size=12)


    i=0
    while i < len(valores):

        #The values inside the loc are boolean, it is a mask
        frequenciac = freqclass.loc[freqclass['classe']>=valores[i]]


        #This is boolean result
        mask = freqclass['classe']>=valores[i]
        #Select only the true values
        width1 = width[mask]

        #Plot
        size = tabresultall.loc[tabresultall['minsize'] == valores[i],:]

        parexp = size.iloc[0,4]
        parpower = size.iloc[1,4]
        parweibull = size.iloc[2,[5,4]]

        parametros = [parexp, parpower, parweibull]

        dbhmin = frequenciac.classe
        dbhmax = dbhmin+width1

        mindbh = dbhmin.min()
        maxdbh = dbhmax.max()

        ##Values per hectare (total of 50 hectares)
        constha = 0.02
        div = np.sum(frequenciac.n)*constha/width1


        plt.scatter(freqclass.classe+(width/2.), (freqclass.n)*constha/width, facecolors='none', edgecolors='k',s=20, linewidths=1, label='_nolegend_')

        plt.plot(frequenciac.classe+(width1/2.), listafunc[j](dbhmin, dbhmax, parametros[j], mindbh, maxdbh)*div, color=colorcoletor[j][i], linewidth=0.8, linestyle=listalinestyle[j], label=nomestitle[i])

        plt.loglog()

        i+=1

    plt.ylim(0.00001,3)
    plt.xlim(1,600)
    plt.legend(loc='best')
    plt.title('Fit '+listanome[j])

    plt.xlabel(r'Canopy disturbance area (m$^{2}$)', labelpad=10)
    plt.ylabel(r'Frequency (n.ha$^{-1}$.m$^{2}$)', labelpad=10)


    plt.savefig('../Exit/gap_size_fits_'+listanome[j]+'.png', dpi=300, bbox_inches='tight')


    plt.close()

    j+=1


########################################################################################################
########################################################################################################
########################################################################################################
##Plot Figure 6d

i=3

while i < len(valores):

    frequenciac = freqclass.loc[freqclass['classe']>=valores[i]]

    #This is boolean result
    mask = freqclass['classe']>=valores[i]

    #Select only the true values
    width1 = width[mask]


    ##Plot
    size = tabresultall.loc[tabresultall['minsize'] == valores[i],:]

    parexp = size.iloc[0,4]
    parpower = size.iloc[1,4]
    parweibull = size.iloc[2,[5,4]]

    dbhmin = frequenciac.classe
    dbhmax = dbhmin+width1

    mindbh = dbhmin.min()
    maxdbh = dbhmax.max()

    ##Values per hectare (total of 50 hectares)
    constha = 0.02
    div = np.sum(frequenciac.n)*constha/width1

    #Plot loglog
    plt.figure(figsize=(4.4, 3.4))
    plt.rc('font', family='Times New Roman', size=12)
    plt.scatter(freqclass.classe+(width/2.), (freqclass.n)*constha/width, facecolors='none', edgecolors='k',s=20, linewidths=1, label='_nolegend_')


    plt.plot(frequenciac.classe+(width1/2.), f.funcpower(dbhmin, dbhmax, parpower, mindbh, maxdbh)*div, color='blue', linewidth=0.8, label='Power')
    plt.plot(frequenciac.classe+(width1/2.), f.funcweibull(dbhmin, dbhmax, parweibull[0],parweibull[1], mindbh, maxdbh)*div, color='red', linewidth=0.8, linestyle='--', label='Weibull') #dashes=(5, 5)

    plt.loglog()
    plt.ylim(0.00001,1.1)
    plt.xlim(1.7,600)
    plt.legend(loc='lower right')
    plt.xlabel(r'Canopy disturbance area (m$^{2}$)', labelpad=10)
    plt.ylabel(r'Frequency (n.ha$^{-1}$.m$^{2}$)', labelpad=10)
    plt.xticks([2,5,10,20,50,100,200,500], ['2','5','10','20','50','100','200','500'])
    # plt.yticks([0.01, 0.1, 1], [ '0.01', '0.1', '1'])

    plt.axvspan(1.7, valores[i], alpha=0.2, color='gray')
    
    # plt.yticks([0.001, 0.01, 0.1, 1], ['0.001', '0.01', '0.1', '1'])
    # plt.ticklabel_format(axis='y', style='sci')


    #Proportion y
    # plt.axvline(x=valores[i], ymin=0.0001, ls='--', color='k', linewidth=0.5, zorder=1) #ymax=valoresylinha[i]


    plt.savefig('../Exit/main_gap_size_fits_log'+nomes[i]+'.png', dpi=300, bbox_inches='tight')

    plt.close()

    i+=1

