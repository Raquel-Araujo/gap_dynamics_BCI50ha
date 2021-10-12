import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt

###EXPONENTIAL
#calcdbhhist=function(dbhmin,dbhmax,fitfcn,param)
###Calculate the probability of each stem occur in each size class given a parameter. 
###Subtration of the cumulative densidy distributions (CDF), resulting a in PDF
#dbhmin = dbh size classes, dbhmax = each size classes values + 1 (class size), param = initial parameter
#mindbh = value of the mininum dbh, maxdbh = value of the maximum dbh

def funcexp(dbhmin, dbhmax, param, mindbh, maxdbh):

    # mindbh = np.min(dbhmin)
    # maxdbh = np.max(dbhmax)

    # print('mindbh:',mindbh)
    # print('maxdbh:',maxdbh)

    probexp = stats.expon.cdf(dbhmax, scale = 1./param) - stats.expon.cdf(dbhmin, scale = 1./param)
    probexptotal = stats.expon.cdf(maxdbh, scale = 1./param) - stats.expon.cdf(mindbh, scale = 1./param)

    probexpnorm = probexp/probexptotal

    return probexpnorm

#errordbhlik=function(param,fitfcn,dbhmin,dbhmax,n,totstems)
###Calculate the negative log of the likelihood

def funclike(param, dbhmin, dbhmax, y, mindbh, maxdbh):

    probexpnorm = funcexp(dbhmin, dbhmax, param, mindbh, maxdbh)
    loglike = -np.sum(y * np.log(probexpnorm))

    return loglike

#fitdbhlik=function(nstem,mindbh,maxdbh,fitfcn)
#fit=optimize(f=errordbhlik,interval=c(-0.5*initpar,2*initpar),fitfcn=fitfcn,dbhmin=mindbh,dbhmax=maxdbh,n=nstem,totstems=totstems)
###Calculate the the minimum value of the negative log likelihood.

def funcfitmle(param, dbhmin, dbhmax, y, mindbh, maxdbh):

    bnds = [(0.0001, 0.3)]
    maxlike = minimize(fun=funclike, x0=param, args=(dbhmin,dbhmax, y, mindbh, maxdbh), method="SLSQP", bounds=bnds)
    
    return (maxlike)

#####################################################
###POWER

# expectedn=dbhmax^(1-param[1])-dbhmin^(1-param[1])
# totexpected=maxdbh^(1-param[1])-mindbh^(1-param[1])

def funcpower(dbhmin, dbhmax, param, mindbh, maxdbh):

    # mindbh = np.min(dbhmin)
    # maxdbh = np.max(dbhmax)

    # print('mindbh:',mindbh)
    # print('maxdbh:',maxdbh)
   
    prob = dbhmax**(1-param)-dbhmin**(1-param)
    probtotal = maxdbh**(1-param)-mindbh**(1-param)

    probnorm = prob/probtotal

    return probnorm

###Calculate the negative log of the likelihood
def funclikepower(param, dbhmin, dbhmax, y, mindbh, maxdbh):

    probnorm = funcpower(dbhmin, dbhmax, param, mindbh, maxdbh)
    loglike = -np.sum(y * np.log(probnorm))

    return loglike

###Calculate the the minimum value of the negative log likelihood.

def funcfitmlepower(param, dbhmin, dbhmax, y, mindbh, maxdbh):

    bnds = [(0.1, 10)]
    maxlike = minimize(fun=funclikepower, x0=param, args=(dbhmin,dbhmax, y, mindbh, maxdbh), method="SLSQP", bounds=bnds)
    
    return (maxlike)

#######################################################
###WEIBULL

#expectedn=pweibull(dbhmax,param[1],param[2])-pweibull(dbhmin,param[1],param[2])
#totexpected=pweibull(maxdbh,param[1],param[2])-pweibull(mindbh,param[1],param[2])

###Calculate the probability of each stem occur in each size class given a parameter. 

def cdfweibull(dbh, pscale, pshape):
    return 1 - np.exp(-(dbh/pscale)**pshape)

def funcweibull(dbhmin, dbhmax, pscale, pshape, mindbh, maxdbh):


    # mindbh = np.min(dbhmin)
    # maxdbh = np.max(dbhmax)

    prob = cdfweibull(dbhmax, pscale, pshape) - cdfweibull(dbhmin, pscale, pshape)
    probtotal = cdfweibull(maxdbh, pscale, pshape) - cdfweibull(mindbh, pscale, pshape)

    probnorm = prob/probtotal

    return probnorm

###Calculate the negative log of the likelihood
def funclikeweibull(par, dbhmin, dbhmax, y, mindbh, maxdbh):

    probnorm = funcweibull(dbhmin, dbhmax, par[0], par[1], mindbh, maxdbh)
    loglike = -np.sum(y * np.log(probnorm))

    return loglike

###Calculate the the minimum value of the negative log likelihood.

def funcfitmleweibull(par, dbhmin, dbhmax, y, mindbh, maxdbh):

    # bnds = [(2.5, 50), (0.001, 2)]
    bnds = [(0.01, 100), (0.001, 2)]

    maxlike = minimize(fun=funclikeweibull, x0=par, args=(dbhmin,dbhmax, y, mindbh, maxdbh), method="SLSQP", bounds=bnds)
    
    return (maxlike)


##########################################################################
###Functions for all groups

###Funcao do bootstrap ajustada para todos os grupos com diferentes dbhmin
def funcbootstraptodos(frequencia):

    print('frenquencia:',frequencia)
    sample = np.random.randint(low = 1, high = 52, size = 51)
    # print('sample:',sample)

    plotsample = frequencia.loc[:,sample]
    # print(plotsample)

    freqtotalsample = plotsample.sum(axis=1)
    # print(freqtotalsample)
    
    return freqtotalsample, frequencia['CDIAM']


def fitmleboots(frequenciaplot, mindbh, group):

    colparamexp = []
    collikeexp = []
    colparampower = []
    collikepower = []
    colparamweibull = []
    colparam1weibull = []
    collikeweibull = []

    i = 0
    while i <= 1000:

        
        freqtotalsample, dbhmin = funcbootstraptodos(frequenciaplot)
        dbhmax = dbhmin + 1
        # print('dbhmin:',dbhmin)
        # print('dbhmax:',dbhmax)

        maxdbh = 118

        #EXPONENTIAL
        param = 0.0984
        maxlikelihoodexp = funcfitmle(param, dbhmin, dbhmax, freqtotalsample, mindbh, maxdbh)
        # print(maxlikelihoodexp)

        colparamexp.append(maxlikelihoodexp.x)
        collikeexp.append(maxlikelihoodexp.fun)

        #POWER
        param = 1.9975
        maxlikelihoodpower = funcfitmlepower(param, dbhmin, dbhmax, freqtotalsample, mindbh, maxdbh)
        # print(maxlikelihoodpower)

        colparampower.append(maxlikelihoodpower.x)
        collikepower.append(maxlikelihoodpower.fun)

        #WEIBULL
        pscale = 10.86
        pshape = 0.93
        par=[pscale, pshape]

        maxlikelihoodweibull = funcfitmleweibull(par, dbhmin, dbhmax, freqtotalsample, mindbh, maxdbh)
        # print(maxlikelihoodweibull)

        colparamweibull.append(maxlikelihoodweibull.x[0])
        colparam1weibull.append(maxlikelihoodweibull.x[1])
        collikeweibull.append(maxlikelihoodweibull.fun)

        i += 1

    ###norm fit faz media e desvio
    mediaparamexp, desvioparamexp = stats.norm.fit(np.array(colparamexp))
    mediaparampower, desvioparampower = stats.norm.fit(np.array(colparampower))
    mediaparamweibull, desvioparamweibull = stats.norm.fit(np.array(colparamweibull))
    mediaparam1weibull, desvioparam1weibull = stats.norm.fit(np.array(colparam1weibull))

    
    # meanparamexp = np.array(colparamexp).mean()
    # meanlikeexp = np.array(collikeexp).mean()
    # meanparampower = np.array(colparampower).mean()
    # meanlikepower = np.array(collikepower).mean()
    # meanparamweibull = np.array(colparamweibull).mean()
    # meanparam1weibull = np.array(colparam1weibull).mean()
    # meanlikeweibull = np.array(collikeweibull).mean()

    # print(meanparamexp, meanlikeexp)
    # print(meanparampower, meanlikepower)
    # print(meanparamweibull, meanparam1weibull, meanlikeweibull)


    ###Calculate confidence intervals for parameters
    icminexpq = np.quantile(colparamexp, 0.025)
    icmaxexpq = np.quantile(colparamexp, 0.975)

    icminpowerq = np.quantile(colparampower, 0.025)
    icmaxpowerq = np.quantile(colparampower, 0.975)

    icminweibullq = np.quantile(colparamweibull, 0.025)
    icmaxweibullq = np.quantile(colparamweibull, 0.975)

    icminweibullq1 = np.quantile(colparam1weibull, 0.025)
    icmaxweibullq1 = np.quantile(colparam1weibull, 0.975)

    print(icminweibullq, icmaxweibullq)

    ic = [icminexpq, icmaxexpq, icminpowerq, icmaxpowerq, icminweibullq, icmaxweibullq, icminweibullq1, icmaxweibullq1]

    media, desvio = [mediaparamexp, mediaparampower, mediaparamweibull, mediaparam1weibull], [desvioparamexp, desvioparampower, desvioparamweibull, desvioparam1weibull]
     

    tabelaresultado = tabela(group, media, desvio, ic) ###call the function tabela


    return tabelaresultado
 


def tabela(group, mediasall, desviosall, ic):

    # tabresult = pd.DataFrame(columns=['group','param1', 'param2', 'loglike', 'aic'])

    tabresultall = pd.DataFrame(columns=['group','param1', 'param2'])

    #Exponential
    tabresultall.loc['exp', ('group','param1')] = group, mediasall[0]
    tabresultall.loc['expstd', ('group','param1')] = group, desviosall[0]
    
    media = np.array(tabresultall.loc['exp', ['param1']].tolist())
    desvio = np.array(tabresultall.loc['expstd', ['param1']].tolist())
    
    tabresultall.loc['icminexp', ('param1')] = ic[0]
    tabresultall.loc['icmaxexp', ('param1')] = ic[1]
    

    #Power
    tabresultall.loc['power', ('group','param1')] = group, mediasall[1]
    tabresultall.loc['powerstd', ('group','param1')] = group, desviosall[1]

    media = np.array(tabresultall.loc['power', ['param1']].tolist())
    desvio = np.array(tabresultall.loc['powerstd', ['param1']].tolist())
    
    tabresultall.loc['icminpower', ('param1')] = ic[2]
    tabresultall.loc['icmaxpower', ('param1')] = ic[3]

    #Weibull
    tabresultall.loc['weibull', ('group','param1','param2')] = group, mediasall[2], mediasall[3]
    tabresultall.loc['weibullstd', ('group','param1','param2')] = group, desviosall[2], desviosall[3]

    media = np.array(tabresultall.loc['weibull', ['param1', 'param2']].tolist())
    desvio = np.array(tabresultall.loc['weibullstd', ['param1','param2']].tolist())
    
    tabresultall.loc['icminweibull', ('param1','param2')] = ic[4], ic[6]
    tabresultall.loc['icmaxweibull', ('param1','param2')] = ic[5], ic[7]

    tabresultall.loc[:,'group'] = np.repeat(group, len(tabresultall.loc[:,'group']))

    return tabresultall



####Funcao para ajuste para todos os grupos, sem bootstrap

def fitmlegroups(frequenciaplot, mindbh, group):

   

    # freqtotalsample = (frequenciaplot.drop('CDIAM', axis=1).sum(axis=1))
    freqtotalsample = (frequenciaplot.drop('classe', axis=1).sum(axis=1))

    dbhmin = frequenciaplot['classe']
    dbhmax = dbhmin + 1
    # print('dbhmin:',dbhmin)
    # print('dbhmax:',dbhmax)

    # maxdbh = 118
    maxdbh = dbhmax.max()

    #EXPONENTIAL
    # param = 0.0984
    param = 0.007
    maxlikelihoodexp = funcfitmle(param, dbhmin, dbhmax, freqtotalsample, mindbh, maxdbh)
    # print(maxlikelihoodexp)

    colparamexp = (maxlikelihoodexp.x[0])
    collikeexp = (maxlikelihoodexp.fun)

    #POWER
    # param = 1.9975
    param = 0.8
    maxlikelihoodpower = funcfitmlepower(param, dbhmin, dbhmax, freqtotalsample, mindbh, maxdbh)
    # print(maxlikelihoodpower)

    colparampower = (maxlikelihoodpower.x[0])
    collikepower = (maxlikelihoodpower.fun)

    #WEIBULL
    # pscale = 10.86
    # pshape = 0.93
    pscale = 1
    pshape = 0.5
    par=[pscale, pshape]

    maxlikelihoodweibull = funcfitmleweibull(par, dbhmin, dbhmax, freqtotalsample, mindbh, maxdbh)
    # print(maxlikelihoodweibull)

    colparamweibull = (maxlikelihoodweibull.x[0])
    colparam1weibull = (maxlikelihoodweibull.x[1])
    collikeweibull = (maxlikelihoodweibull.fun)

    
    parametro = [colparamexp, colparampower, colparamweibull, colparam1weibull] 
    like = [collikeexp, collikepower, collikeweibull]

    k = [1,1,2] 
    aic = 2*np.array(k)+2*(np.array(like))

    print(parametro)
    print(like)
    print(aic)
    

    tabelaresultado = tabelamlegroups(group, parametro, like, aic) ###call the function tabela
    
    return tabelaresultado



def tabelamlegroups(group, parametro, like, aic):

    tabresult = pd.DataFrame(columns=['group','param1', 'param2', 'loglike', 'aic'])

    
    #Exponential
    tabresult.loc['exp', ('group','param1', 'loglike', 'aic')] = group, parametro[0], like[0], aic[0]
    
    #Power
    tabresult.loc['power', ('group','param1', 'loglike', 'aic')] = group, parametro[1], like[1], aic[1]

    #Weibull
    tabresult.loc['weibull', ('group','param1','param2','loglike', 'aic')] = group, parametro[2], parametro[3], like[2], aic[2]


    return tabresult



def funcweibullmod(dbhmin, dbhmax, parametros, mindbh, maxdbh):

    pscale, pshape = parametros[0], parametros[1]

    # mindbh = np.min(dbhmin)
    # maxdbh = np.max(dbhmax)

    prob = cdfweibull(dbhmax, pscale, pshape) - cdfweibull(dbhmin, pscale, pshape)
    probtotal = cdfweibull(maxdbh, pscale, pshape) - cdfweibull(mindbh, pscale, pshape)

    probnorm = prob/probtotal

    return probnorm
