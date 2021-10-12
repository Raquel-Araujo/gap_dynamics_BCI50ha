###Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import chisquare
import scipy.stats as stats
import statsmodels.api as sm
from scipy import stats


###EXPONENCIAL
###Funcao para o ajuste de distribuicao exponencial

def funcexp(x, a, b):
    return a * np.exp(b * x)

def exponencial(x,y,p0,nomeclasse):
   
    popt, pcov = curve_fit(funcexp, x, y, p0)
    # print(popt)
    # print(pcov)

    ###Separando os parametros
    a = popt[0] 
    b = popt[1]
    c='-'
    d='-'
    # print(a)
    # print(b)

    media = np.mean(y)
    print('Media = ', media)
    n = len(y)
    # print(n)
    p = len(popt)
    # print(p)
    k = 1
    ypredict = funcexp(x, a, b)
    sqt = np.sum(np.power(y-media, 2))
    sqr = np.sum(np.power(y-ypredict, 2))
    qmr = sqr/(n-p)
    desvio = np.sqrt(qmr)
    # print(desvio)
    erropadrao = desvio/np.sqrt(n)

    #Desvio padrao medio percentual Syx%
    syxporc = (erropadrao/media)*100
    # print(syxporc)

    #R2 para modelos nao lineares
    r2 = 1-(sqr/sqt)
    # r2 = r2_score(y, ypredict) essa funcao produz o mesmo resultado da formula
    # print(r2)

    #R2 ajustado
    r2adj = 1-((n-1)/(n-p))*(sqr/sqt)
    # print(r2adj)

    #AIC
    aic = n * np.log(sqr/n) + 2*k
    # print(aic)

    #Qui-quadrado
    quicalc = np.sum((np.power(y-ypredict, 2))/ypredict)
    # print('quiquad calculado', quicalc)
    quitab = stats.chi2.ppf(q = 0.95, df = n-1)
    # print('quiquad tabelado', quitab)
    quiquad = chisquare(y,ypredict)
    # print(quiquad)
    qui_valorp = 1 - stats.chi2.cdf(x=quicalc,df=n-1)
    # print(qui_valorp)

    # print('y = ', y)
    # print('yest = ', ypredict)
    
    #Model
    model = 'a*exp(b*x)'
   
    ###Fazer dataframe com os parametros...
    sumario= pd.DataFrame([model, a, b, c, d, r2, r2adj, syxporc, aic, qui_valorp ], 
                                    index = ['Model', 'a', 'b', 'c', 'd', 'R2', 'R2adj', 'Syx%','AIC', 'Chi_valorp'])

    ###Exportar sumario
    sumario.to_csv('../Saida/sumario_'+nomeclasse+'.csv')

    ###Plot
    plt.scatter(x,y,c=[0,0,0,0.5], facecolors='none', edgecolors='k', s=20, linewidths=0.5)
    plt.plot(x, ypredict, color='silver', linestyle='--', linewidth=1.5, dashes=(8,5))
    plt.savefig('../Saida/plot_'+nomeclasse+'.png', figsize=(4, 4), dpi=300)
    plt.close()
    return sumario, ypredict,[a,b]

    #################################################################################



###POWER
###Funcao para o ajuste de distribuicao potencial (Power)

def funcpower(x, a, b):
    return a * np.power(x,b)

def power(x,y,p0,nomeclasse):
   
    popt, pcov = curve_fit(funcpower, x, y, p0)
    # print(popt)
    # print(pcov)

    ###Separando os parametros
    a = popt[0] 
    b = popt[1]
    c='-'
    d='-'
    # print(a)
    # print(b)

    media = np.mean(y)
    # print('Media = ', media)
    n = len(y)
    # print('n = ',n)
    p = len(popt)
    # print('p eh igual a ', p)
    k = 1
    ypredict = funcpower(x, a, b)
    sqt = np.sum(np.power(y-media, 2))
    sqr = np.sum(np.power(y-ypredict, 2))
    qmr = sqr/(n-p)
    desvio = np.sqrt(qmr)
    # print(desvio)
    erropadrao = desvio/np.sqrt(n)

    #Desvio padrao medio percentual Syx%
    syxporc = (erropadrao/media)*100
    # print(syxporc)

    #R2
    r2 = 1-(sqr/sqt)
    # r2score = r2_score(y, ypredict) #essa funcao produz o mesmo resultado da formula
    # print(r2)
    # print('r2score', r2)

    #R2 ajustado
    r2adj = 1.-((n-1.)/(n-p))*(1.-r2)
    # print(r2adj)

    #AIC
    aic = n * np.log(sqr/n) + 2*k
    # print(aic)

    #Qui-quadrado
    quicalc = np.sum((np.power(y-ypredict, 2))/ypredict)
    # print('quiquad calculado', quicalc)
    quitab = stats.chi2.ppf(q = 0.95, df = n-1)
    # print('quiquad tabelado', quitab)
    quiquad = chisquare(y,ypredict)
    # print(quiquad)
    qui_valorp = 1 - stats.chi2.cdf(x=quicalc,df=n-1)
    # print(qui_valorp)

    # print('y = ', y)
    # print('yest = ', ypredict)
    
    #Model
    model = 'a*x**b'

    ###Fazer dataframe com os parametros...
    sumario= pd.DataFrame([model, a, b, c, d, r2, r2adj, syxporc, aic, qui_valorp ], 
                                    index = ['Model', 'a', 'b', 'c', 'd', 'R2', 'R2adj', 'Syx%','AIC', 'Chi_valorp'])

    ###Exportar sumario
    sumario.to_csv('../Saida/sumario_'+nomeclasse+'.csv')

    ###Plot
    plt.scatter(x,y,c=[0,0,0,0.5], facecolors='none', edgecolors='k', s=20, linewidths=0.5)
    plt.plot(x, ypredict, color='silver', linestyle='--', linewidth=1.5, dashes=(8,5))
    plt.savefig('../Saida/plot_'+nomeclasse+'.png', figsize=(4, 4), dpi=300)
    plt.close()
    return sumario, ypredict, [a,b]

    #################################################################################



###WEIBULL
###Funcao para o ajuste de distribuicao de Weibull

def funcweibull(x, a, b, c):
    return 1-np.exp(-1*((x-a)/b)**c)

def weibull(x,y,nomeclasse):

    ###Como a funcao eh a acumulada de Weibull, eh preciso deixar os dados de entrada em frequencia acumulada
    #..e o x como limites de classe (ao inves de centro de classe)
    yacum = np.cumsum(y)
    # print(yacum)
    yprob = yacum/yacum[-1] #dividindo pelo total
    ynovo = np.concatenate(([0], yprob)) #acrescentando o zero
    # print('ynovo =',ynovo)
    # print('xmax', np.max(x))
    xnovo = np.arange(10, np.max(x)+5, 5)
    # print('xnovo =',xnovo)

    popt, pcov = curve_fit(funcweibull, xnovo, ynovo, maxfev=100000)
    # print(popt)
    # print(pcov)

    ###Separando os parametros
    a = popt[0] 
    b = popt[1]
    c = popt[2]
    d='-'
    # print(a)
    # print(b)

    media = np.mean(y)
    # print('Media = ', media)
    n = len(y)
    # print(n)
    p = len(popt)
    # print(p)
    k = 1
    
    ypred_acum = funcweibull(xnovo, a, b, c)

    ###Invertendo a frequencia acumulada e deixando novamente no formato J (multiplicando pelo total)
    ypredict = (np.diff(ypred_acum))*619.5
    # print(ypredict)
    
    sqt = np.sum(np.power(y-media, 2))
    sqr = np.sum(np.power(y-ypredict, 2))
    qmr = sqr/(n-p)
    desvio = np.sqrt(qmr)
    # print(desvio)
    erropadrao = desvio/np.sqrt(n)

    #Desvio padrao medio percentual Syx%
    syxporc = (erropadrao/media)*100
    # print(syxporc)

    #R2
    r2 = 1-(sqr/sqt)
    # r2 = r2_score(y, ypredict) essa funcao produz o mesmo resultado da formula
    # print(r2)

    #R2 ajustado
    r2adj = 1.-((n-1.)/(n-p))*(1.-r2)
    # print(r2adj)

    #AIC
    aic = n * np.log(sqr/n) + 2*k
    # print(aic)


    #Qui-quadrado
    quicalc = np.sum((np.power(y-ypredict, 2))/ypredict)
    # print('quiquad calculado', quicalc)
    quitab = stats.chi2.ppf(q = 0.95, df = n-1)
    # print('quiquad tabelado', quitab)
    quiquad = chisquare(y,ypredict)
    # print(quiquad)
    qui_valorp = 1 - stats.chi2.cdf(x=quicalc,df=n-1)
    # print(qui_valorp)

    # print('y = ', y)
    # print('yest = ', ypredict)
    
    #Model
    model = '1-exp(-1*((x-a)/b)**c)'

    ###Fazer dataframe com os parametros...
    sumario= pd.DataFrame([model, a, b, c, d, r2, r2adj, syxporc, aic, qui_valorp ], 
                                    index = ['Model', 'a', 'b', 'c', 'd', 'R2', 'R2adj', 'Syx%','AIC', 'Chi_valorp'])

    ###Exportar sumario
    sumario.to_csv('../Saida/sumario_'+nomeclasse+'.csv')

    ###Plot
    plt.scatter(x,y,c=[0,0,0,0.5], facecolors='none', edgecolors='k', s=20, linewidths=0.5)
    plt.plot(x, ypredict, color='silver', linestyle='--', linewidth=1.5, dashes=(8,5))
    plt.savefig('../Saida/plot_'+nomeclasse+'.png', figsize=(4, 4), dpi=300)
    plt.close()
    return sumario, ypredict, [a,b,c]

    #################################################################################



###LOG-LOG
###Funcao para o ajuste de distribuicao logaritmica

def funclinear(x, a, b):
    return b*x+a

def log(x,y,nomeclasse):

    ###Variaveis log
    xnovo = np.log10(x)
    ynovo = np.log10(y)
    # print('xnovo = ',xnovo)
    # print('ynovo = ',ynovo)
    

    popt, pcov = curve_fit(funclinear, xnovo, ynovo)
    # print(popt)
    # print(pcov)

    ###Separando os parametros
    a = popt[0] 
    b = popt[1]
    c='-'
    d='-'
    # print(a)
    # print(b)

    ###Usa os parametros do yoriginal pra poder comparar com os outros modelos
    media = np.mean(y)
    # print('Media = ', media)
    n = len(y)
    # print(n)
    p = len(popt)
    # print(p)
    k = 1

    ###Fazendo o inverso do log (10 elevado a funcao) para deslogaritimizar a funcao
    ypredict = np.power(10,funclinear(xnovo, a, b))

    sqt = np.sum(np.power(y-media, 2))
    sqr = np.sum(np.power(y-ypredict, 2))
    qmr = sqr/(n-p)
    desvio = np.sqrt(qmr)
    # print(desvio)
    erropadrao = desvio/np.sqrt(n)

    #Desvio padrao medio percentual Syx%
    syxporc = (erropadrao/media)*100
    # print(syxporc)

    #R2
    r2 = 1-(sqr/sqt)
    # r2 = r2_score(y, ypredict) essa funcao produz o mesmo resultado da formula
    # print(r2)

    #R2 ajustado
    r2adj = 1.-((n-1.)/(n-p))*(1.-r2)
    # print(r2adj)

    #AIC
    aic = n * np.log(sqr/n) + 2*k
    # print(aic)

    #Qui-quadrado
    quicalc = np.sum((np.power(y-ypredict, 2))/ypredict)
    # print('quiquad calculado', quicalc)
    quitab = stats.chi2.ppf(q = 0.95, df = n-1)
    # print('quiquad tabelado', quitab)
    quiquad = chisquare(y,ypredict)
    # print(quiquad)
    qui_valorp = 1 - stats.chi2.cdf(x=quicalc,df=n-1)
    # print(qui_valorp)

    # print('y = ', y)
    # print('yest = ', ypredict)
    
    #Model
    model = 'a+bx'

    ###Fazer dataframe com os parametros...
    sumario= pd.DataFrame([model, a, b, c, d, r2, r2adj, syxporc, aic, qui_valorp ], 
                                    index = ['Model', 'a', 'b', 'c', 'd','R2', 'R2adj', 'Syx%','AIC', 'Chi_valorp'])

    ###Exportar sumario
    sumario.to_csv('../Saida/sumario_'+nomeclasse+'.csv')

    ###Plot
    plt.scatter(x,y,c=[0,0,0,0.5], facecolors='none', edgecolors='k', s=20, linewidths=0.5)
    plt.plot(x, ypredict, color='silver', linestyle='--', linewidth=1.5, dashes=(8,5))
    plt.savefig('../Saida/plot_'+nomeclasse+'.png', figsize=(4, 4), dpi=300)
    plt.close()
    return sumario, ypredict, [a,b]

    #################################################################################




###SIGMOIDE
###Funcao para o ajuste de distribuicao sigmoide

def funcsigmoide(x, a, b):
    return 1/(1+np.exp(-1*(a+b*x)))

def sigmoide(x,y,nomeclasse):

    ###Como a funcao eh a acumulada de Weibull, eh preciso deixar os dados de entrada em frequencia acumulada
    #..e o x como limites de classe (ao inves de centro de classe)
    yacum = np.cumsum(y)
    # print(yacum)
    yprob = yacum/yacum[-1] #dividindo pelo total
    ynovo = np.concatenate(([0], yprob)) #acrescentando o zero
    # print(ynovo)
    # print('xmax', np.max(x))
    xnovo = np.arange(10, np.max(x)+5, 5)
    # print(xnovo)


    popt, pcov = curve_fit(funcsigmoide, xnovo, ynovo)
    # print(popt)
    # print(pcov)

    ###Separando os parametros
    a = popt[0] 
    b = popt[1]
    c='-'
    d='-'
    # print(a)
    # print(b)

    media = np.mean(y)
    # print('Media = ', media)
    n = len(y)
    # print(n)
    p = len(popt)
    # print(p)
    k = 1
    
    ypred_acum = funcsigmoide(xnovo, a, b)

    ###Invertendo a frequencia acumulada e deixando novamente no formato J (multiplicando pelo total)
    ypredict = (np.diff(ypred_acum))*619.5
    # print(ypredict)
    
    sqt = np.sum(np.power(y-media, 2))
    sqr = np.sum(np.power(y-ypredict, 2))
    qmr = sqr/(n-p)
    desvio = np.sqrt(qmr)
    # print(desvio)
    erropadrao = desvio/np.sqrt(n)

    #Desvio padrao medio percentual Syx%
    syxporc = (erropadrao/media)*100
    # print(syxporc)

    #R2
    r2 = 1-(sqr/sqt)
    # r2 = r2_score(y, ypredict) essa funcao produz o mesmo resultado da formula
    # print(r2)

    #R2 ajustado
    r2adj = 1.-((n-1.)/(n-p))*(1.-r2)
    # print(r2adj)

    #AIC
    aic = n * np.log(sqr/n) + 2*k
    # print(aic)


    #Qui-quadrado
    quicalc = np.sum((np.power(y-ypredict, 2))/ypredict)
    # print('quiquad calculado', quicalc)
    quitab = stats.chi2.ppf(q = 0.95, df = n-1)
    # print('quiquad tabelado', quitab)
    quiquad = chisquare(y,ypredict)
    # print(quiquad)
    qui_valorp = 1 - stats.chi2.cdf(x=quicalc,df=n-1)
    # print(qui_valorp)

    # print('y = ', y)
    # print('yest = ', ypredict)
    
    #Model
    model = '1/(1+exp(-1*(a+b*x)))'

    ###Fazer dataframe com os parametros...
    sumario= pd.DataFrame([model, a, b, c, d, r2, r2adj, syxporc, aic, qui_valorp ], 
                                    index = ['Model', 'a', 'b', 'c', 'd', 'R2', 'R2adj', 'Syx%','AIC', 'Chi_valorp'])

    ###Exportar sumario
    sumario.to_csv('../Saida/sumario_'+nomeclasse+'.csv')

    ###Plot
    plt.scatter(x,y,c=[0,0,0,0.5], facecolors='none', edgecolors='k', s=20, linewidths=0.5)
    plt.plot(x, ypredict, color='silver', linestyle='--', linewidth=1.5, dashes=(8,5))
    plt.savefig('../Saida/plot_'+nomeclasse+'.png', figsize=(4, 4), dpi=300)
    plt.close()
    return sumario, ypredict, [a,b]

    #################################################################################



###SIGMOIDE SIMETRICA
###Funcao para o ajuste de distribuicao sigmoide simetrica

def funcsigmoide2(x, a, b, c, d):
    return d+((a-d)/(1+(x/c)**b))

def sigmoide2(x,y,nomeclasse):
   
    popt, pcov = curve_fit(funcsigmoide2, x, y)
    # print(popt)
    # print(pcov)

    ###Separando os parametros
    a = popt[0] 
    b = popt[1]
    c = popt[2]
    d = popt[3]
   

    media = np.mean(y)
    # print('Media = ', media)
    n = len(y)
    # print('n variaveis = ',n)
    p = len(popt)
    # print(p)
    k = 1


    ypredict = funcsigmoide2(x, a, b, c, d)

    sqt = np.sum(np.power(y-media, 2))
    sqr = np.sum(np.power(y-ypredict, 2))
    qmr = sqr/(n-p)
    desvio = np.sqrt(qmr)
    # print(desvio)
    erropadrao = desvio/np.sqrt(n)

    #Desvio padrao medio percentual Syx%
    syxporc = (erropadrao/media)*100
    # print(syxporc)

    #R2
    r2 = 1-(sqr/sqt)
    # r2 = r2_score(y, ypredict) essa funcao produz o mesmo resultado da formula
    # print(r2)

    #R2 ajustado
    r2adj = 1.-((n-1.)/(n-p))*(1.-r2)
    # print(r2adj)

    #AIC
    aic = n * np.log(sqr/n) + 2*k
    # print(aic)

    #Qui-quadrado
    quicalc = np.sum((np.power(y-ypredict, 2))/ypredict)
    # print('quiquad calculado', quicalc)
    quitab = stats.chi2.ppf(q = 0.95, df = n-1)
    # print('quiquad tabelado', quitab)
    quiquad = chisquare(y,ypredict)
    # print(quiquad)
    qui_valorp = 1 - stats.chi2.cdf(x=quicalc,df=n-1)
    # print(qui_valorp)

    # print('y = ', y)
    # print('yest = ', ypredict)
    
    #Model
    model = 'd+((a-d)/(1+(x/c)**b))'

    ###Fazer dataframe com os parametros...
    sumario= pd.DataFrame([model, a, b, c, d, r2, r2adj, syxporc, aic, qui_valorp ], 
                                    index = ['Model', 'a', 'b', 'c', 'd', 'R2', 'R2adj', 'Syx%','AIC', 'Chi_valorp'])

    ###Exportar sumario
    sumario.to_csv('../Saida/sumario_'+nomeclasse+'.csv')

    ###Plot
    plt.scatter(x,y,c=[0,0,0,0.5], facecolors='none', edgecolors='k', s=20, linewidths=0.5)
    plt.plot(x, ypredict, color='silver', linestyle='--', linewidth=1.5, dashes=(8,5))
    plt.savefig('../Saida/plot_'+nomeclasse+'.png', figsize=(4, 4), dpi=300)
    plt.close()
    return sumario, ypredict, [a,b,c,d]

    #################################################################################



###################################################################################
###################################################################################
###################################################################################
###################################################################################

###FUNCAO MODULADA
###Aplica todas as funcoes (exponencial, power, weibull, etc) aas classes de dossel 
###...subdossel e todos. 


###################################################################################
###Ajustes para todos os individuos

def funcmodulo(classe,freqgrupo,nomegrupo,testeweibull=1):  

    frame = []
    frame_yest = []
    frame_coefs = []

    ###Ajuste exponencial
    # print('Ajuste exponencial...')
    x = np.array(classe)
    ytodos = np.array(freqgrupo)
    # print(x)
    # print(ytodos)

    ###Chamando a funcao exponencial que retorna com os parametros e grafico do ajuste 
    p0 = (1654, -0.1)
    sumario_exp, yest_exp, coefs  = exponencial(x,ytodos,p0,(nomegrupo+'_exp'))
    # print('Sumario '+nomegrupo+' = ', sumario_exp)
    # print('Ypredict '+nomegrupo+' = ', yest_exp)
    frame.append(sumario_exp)
    frame_yest.append(yest_exp)
    frame_coefs.append(coefs)


    ###Ajuste power
    # print('Ajuste power...')
    ###Chamando a funcao power que retorna com os parametros e grafico do ajuste 
    p0 = (1654, -0.1)
    sumario_power, yest_power, coefs  = power(x,ytodos,p0,(nomegrupo+'_power'))
    # print('Sumario_power = ', sumario_power)
    # print('Ypredict_power = ', yest_power)
    frame.append(sumario_power)
    frame_yest.append(yest_power)
    frame_coefs.append(coefs)


    ###Ajuste weibull
    # print('Ajuste weibull...')
    ###Chamando a funcao weibull que retorna com os parametros e grafico do ajuste 
    # p0 = (10, 11, 0.98)
    if testeweibull == 1:
        sumario_weibull, yest_weibull, coefs  = weibull(x,ytodos,(nomegrupo+'_weibull'))
        # print('Sumario_weibull = ', sumario_weibull)
        # print('Ypredict_weibull = ', yest_weibull)
        frame.append(sumario_weibull)
        frame_yest.append(yest_weibull)
        frame_coefs.append(coefs)

    ###Ajuste log
    # print('Ajuste log...')
    ###Chamando a funcao weibull que retorna com os parametros e grafico do ajuste 
    sumario_log, yest_log, coefs  = log(x,ytodos,(nomegrupo+'_log'))
    # print('Sumario_log = ', sumario_log)
    # print('Ypredict_log = ', yest_log)
    frame.append(sumario_log)
    frame_yest.append(yest_log)
    frame_coefs.append(coefs)

    ###Ajuste sigmoide
    # print('Ajuste sigmoide...')
    ###Chamando a funcao weibull que retorna com os parametros e grafico do ajuste 
    sumario_sigmoide, yest_sigmoide, coefs  = sigmoide(x,ytodos,(nomegrupo+'_sigmoide'))
    # print('Sumario_sigmoide = ', sumario_sigmoide)
    # print('Ypredict_sigmoide = ', yest_sigmoide)
    frame.append(sumario_sigmoide)
    frame_yest.append(yest_sigmoide)
    frame_coefs.append(coefs)

    ###Ajuste sigmoide simetrica
    # print('Ajuste sigmoide simetrica...')
    ###Chamando a funcao weibull que retorna com os parametros e grafico do ajuste 
    sumario_sigmoide2, yest_sigmoide2, coefs  = sigmoide2(x,ytodos,(nomegrupo+'_sigmoide2'))
    # print('Sumario_sigmoide2 = ', sumario_sigmoide2)
    # print('Ypredict_sigmoide2 = ', yest_sigmoide2)
    frame.append(sumario_sigmoide2)
    frame_yest.append(yest_sigmoide2)
    frame_coefs.append(coefs)

    sumario_todos = pd.concat(frame, axis=1)

    if testeweibull == 1:
        sumario_todos.columns = ['exponencial', 'potencial', 'weibull', 'log-log', 'sigmoide', 'sigmoide2']
    if testeweibull == 0:
        sumario_todos.columns = ['exponencial', 'potencial', 'log-log', 'sigmoide', 'sigmoide2']


    ###Exportar sumario
    sumario_todos.to_csv('../Saida/sumario'+nomegrupo+'.csv')

    return frame_yest, frame_coefs





###################################################################################
###################################################################################
###################################################################################
###################################################################################

###LINEAR
###Funcao para o ajuste linear

def funclinear(x, a, b):
    return b*x+a

def linear(x, y, nomeclasse):


    ###Regressao linear OLS
    xconst = sm.add_constant(x)

    model = sm.OLS(y, xconst, missing = 'drop').fit()
    ypredict = model.predict()
    # print('x = ',x)
    # print('y = ',y)
    # print('ypredict =', ypredict) 

    sumariomodel= model.summary()
    # print(sumariomodel)
    dir(model)
    # print('Parameters: ', dir(model))
    valorp = model.f_pvalue
    # print('Valor-p = ', valorp)
    a = model.params[0]
    b = model.params[1]
    r2 = model.rsquared

    r = stats.pearsonr(x, y)
        
    ynovo = pd.Series(y).dropna().tolist()

    # print('ynovo = ',ynovo)

    media = np.mean(ynovo)
    # print('Media = ', media)
    n = len(ynovo)
    # print(n)
    p = len(model.params)
    # print(p)
    modelo = 'a+bx'


    sqt = np.sum(np.power(ynovo-media, 2))
    sqr = np.sum(np.power(ynovo-ypredict, 2))
    qmr = sqr/(n-p)
    desvio = np.sqrt(qmr)
    # print(desvio)
    erropadrao = desvio/np.sqrt(n)
    syxporc = (erropadrao/media)*100


    ##Fazer dataframe com os parametros...
    sumario= pd.DataFrame([modelo,a, b, r2, syxporc, valorp, n, r[0]], 
                                    index = ['model','a', 'b', 'R2', 'Syx%', 'pvalue', 'n', 'r'])
    # sumario.round(3)
    ###Exportar sumario
    sumario.to_csv('../Exit/sumario_'+nomeclasse+'.csv')

    ###Plot
    # plt.scatter(x,y,c=[0,0,0,0.5], facecolors='none', edgecolors='k', s=20, linewidths=0.5)
    # plt.plot(x, ypredict, color='silver', linestyle='--', linewidth=1.5, dashes=(8,5))
    # plt.savefig('../Saida/plot_'+nomeclasse+'.png', figsize=(4, 4), dpi=300)
    # plt.show()
    # return [a, b, r2, syxporc, valorp, n]
    return sumario

    #################################################################################

