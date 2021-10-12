# code to fit size distributions using exact values of sizes rather than frequencies within bins
# (note that for dbh data, there is always a measurement precision, and bins at the size of the
# measurement precision are the best way to do the fits)

# note - using here the Pareto distribution for the power function pdf  
# the location parameter is the minimum value under the pdf
# the exponent in the power function is equal to the (-shape-1) 

# still need to normalize for truncated upper and lower bounds

rm(list=ls())
library(EnvStats) # for the pareto distribution - but log option not available 
#library(actuar) # for the Pareto (power function) probability distribution, but only available for R version > 4,
# need to get rid of antivirus block on internet.dll before R 4.1.1 can be run
library(dplyr)

setfitsizedistcontdata <- function(sizedf,minsize=c(2,5,10,25),maxsize=c(720,900,NA),fitfcn=c("exp","pow","weib")) {
  # fits size distributions to continuous data
  # INPUT VARIABLES 
  # sizedf is a dataframe with one row per individual, and three columns:
  #    id is the unique id of the individual (not used)
  #    size is the size of the individual to be used in the size distribution fitting
  #    bsunit is the unit to which the individual belongs, where the unit is used for bootstrapping
  # for the current application to fitting canopy disturbances, 
  #    size is in m2, and bsunit is the date of the interval when the disturbance was recorded.  
  # minsize are the minimum sizes for which the function is being fitted (multiple options allowed)
  #       (must be positive for the power function distribution;zero or positive for the others)
  #       This is generally the minimum size measurable or well-measured in the dataset.  
  # maxsize are the maximum sizes for the function being fitted (multiple options allowed)
  #       This can be NA, meaning the function is fit assuming no upper bound on possible values.
  #       This can be set to the maximum size observed or observable in the data.  
  # fitfcn is a character string specifying the function to fit (multiple options allowed)
  #       "weib" for weibull, "exp" for exponential, "pow" for power
  #       Note that the "power function" distribution is the Pareto distribution, with a change of parameters.
  # one fit is done for every possible combination of minsize, maxsize, and fitfcn
  
  for (k in 1:length(maxsize)) {
    for (j in 1:length(minsize)) {
      for (i in 1:length(fitfcn)) {
        usesizedf <- sizedf[sizedf$size>=minsize[j],]
        if (!is.na(maxsize[k])) usesizedf <- usesizedf[usesizedf$size<=maxsize[k],]
        thisfullfit <- fitsizedistcontdata(usesizedf$size,minsize[j],maxsize[k],fitfcn[i])
        if (i==1) thissetfit <- thisfullfit else thissetfit <- rbind(thissetfit,thisfullfit)
      }
      thissetfit$aicdif <- thissetfit$aic-min(thissetfit$aic)
      if (j==1 & k==1) allfits <- thissetfit else allfits <- rbind(allfits,thissetfit)
    }
  }
  return(allfits)
}  # end setfitsizedistnobins


bsfitsizedistcontdata <- function(sizedf,minsize=c(2,5,10,25),maxsize=c(720,900,NA),fitfcn=c("exp","pow","weib"),
                                  nbootstraps=1000,alpha1=0.05, alpha2=0.01) {
  # as above fits size distributions to continuous data, with addition of
  # producing bootstrapped confidence intervals (this is SLOW!)
  # new input variables: 
  #   nbootstraps is the number of bootstrap iteractions to run to get CIs
  #   alpha1 and alpha2 are the p-values at which to calculate CIs
  
  # fits to full data
  for (k in 1:length(maxsize)) {
    for (j in 1:length(minsize)) {
      usesizedf <- sizedf[sizedf$size>=minsize[j],]
      if (!is.na(maxsize[k])) usesizedf <- usesizedf[usesizedf$size<=maxsize[k],]
      for (i in 1:length(fitfcn)) {
        thisfullfit <- fitsizedistcontdata(usesizedf$size,minsize[j],maxsize[k],fitfcn[i])
        if (i==1) thissetfit <- thisfullfit else thissetfit <- rbind(thissetfit,thisfullfit)
      }
      thissetfit$aicdif <- thissetfit$aic-min(thissetfit$aic)
      if (j==1 & k==1) allfits <- thissetfit else allfits <- rbind(allfits,thissetfit)
    }
  }
  
  # now for bootstrapping...
  allfits$nbootstraps <- nbootstraps
  allfits$alpha1 <- alpha1
  allfits$alpha2 <- alpha2
  allfits$par1lo1 <- NA
  allfits$par1hi1 <- NA
  allfits$par1lo2 <- NA
  allfits$par1hi2 <- NA
  allfits$par2lo1 <- NA
  allfits$par2hi1 <- NA
  allfits$par2lo2 <- NA
  allfits$par2hi2 <- NA

  bsunits <- unique(sizedf$bsunit)
  nbsunits <- length(bsunits)
  bsfits <- list()

  q <- 1 # row number in results file 
  for (k in 1:length(maxsize)) {
    for (j in 1:length(minsize)) {
      usesizedf <- sizedf[sizedf$size>=minsize[j],]
      if (!is.na(maxsize[k])) usesizedf <- usesizedf[usesizedf$size<=maxsize[k],]
      for (i in 1:length(fitfcn)) {
        for (m in 1:nbootstraps) {
            thisbsdf <- data.frame(bsunit=sample(bsunits,nbsunits,replace=T))
            thisbsdf %>% count(bsunit) -> unitfreq
            for (n in 1:nrow(unitfreq)) {
              thisdat <- usesizedf$size[usesizedf$bsunit==unitfreq$bsunit[n]]
              thisdat <- rep(thisdat,unitfreq$n[n])
              if (n==1) bsdat<-thisdat else bsdat <-c(bsdat,thisdat)
            }
            thisfit <- fitsizedistcontdata(bsdat,minsize[j],maxsize[k],fitfcn[i])
            if (m==1) setbsfit <- thisfit else setbsfit <- rbind(setbsfit,thisfit)
        }
          # get quantiles of fitted parameter values 
        allfits$par1lo1[q] <- quantile(setbsfit$par1,alpha1/2)
        allfits$par1hi1[q] <- quantile(setbsfit$par1,1-alpha1/2)
        allfits$par1lo2[q] <- quantile(setbsfit$par1,alpha2/2)
        allfits$par1hi2[q] <- quantile(setbsfit$par1,1-alpha2/2)
        if (fitfcn[i]=="weib") {
          allfits$par2lo1[q] <- quantile(setbsfit$par2,alpha1/2)
          allfits$par2hi1[q] <- quantile(setbsfit$par2,1-alpha1/2)
          allfits$par2lo2[q] <- quantile(setbsfit$par2,alpha2/2)
          allfits$par2hi2[q] <- quantile(setbsfit$par2,1-alpha2/2)
        }
        q <- q+1
      }
    }
  }
  
    
  return(allfits)
}  # end bsfitsizedistnobins






################################################
fitsizedistcontdata <- function(sizes,minsize,maxsize,fitfcn)
  # sizes is a vector of the sizes of individuals 
  # fitfcn is a character string specifying the function to fit: "weib" for weibull, "exp" for exponential, "pow" for power
   ############################################
{
  if (fitfcn=="weib") {
    fit=optim(par=c(0.1,1),fn=errordbhlikcont,fitfcn=fitfcn,minsize=minsize,maxsize=maxsize,sizes=sizes,method="SANN")
    fit=optim(par=fit$par,fn=errordbhlikcont,fitfcn=fitfcn,minsize=minsize,maxsize=maxsize,sizes=sizes)
    thisks <- calcksstat(fit$par,sizes,minsize,maxsize,fitfcn)
  }
  else if (fitfcn=="pow") {  # need to use a different function for 1-dimensional optimization
    fit=optimize(f=errordbhlikcont,interval=c(0.001,10),fitfcn=fitfcn,minsize=minsize,maxsize=maxsize,sizes=sizes)
    thisks <- calcksstat(fit$minimum,sizes,minsize,maxsize,fitfcn)
  }
  else if (fitfcn=="exp") {  # need to use a different function for 1-dimensional optimization
    initpar=0.1
    loglikes=calclogliksobssizes(initpar,sizes,minsize,maxsize,fitfcn)
    minloglik=min(loglikes)
    while(minloglik==-Inf) {
      initpar=initpar/2
      loglikes=calclogliksobssizes(initpar,sizes,minsize,maxsize,fitfcn)
      minloglik=min(loglikes)
    }
    fit=optimize(f=errordbhlikcont,interval=c(-0.5*initpar,2*initpar),fitfcn=fitfcn,minsize=minsize,maxsize=maxsize,
                 sizes=sizes)
    thisks <- calcksstat(fit$minimum,sizes,minsize,maxsize,fitfcn)
  }
  
  if (fitfcn=="weib")
    dbhfit=data.frame(minsize=minsize,maxsize=maxsize,fitfcn=fitfcn,npar=2,par1=fit$par[1],par2=fit$par[2],
                      convergerror=fit$convergence,ksstat=thisks,loglike=-fit$value)
  else if (fitfcn=="pow")
    dbhfit=data.frame(minsize=minsize,maxsize=maxsize,fitfcn=fitfcn,npar=1,par1=1+fit$minimum,par2=NA,
                      convergerror=NA,ksstat=thisks,loglike=-fit$objective)
  else if (fitfcn=="exp")
    dbhfit=data.frame(minsize=minsize,maxsize=maxsize,fitfcn=fitfcn,npar=1,par1=fit$minimum,par2=NA,
                      convergerror=NA,ksstat=thisks,loglike=-fit$objective)
  
  dbhfit$aic <- 2*dbhfit$npar - 2*dbhfit$loglike
  return(dbhfit)
} # end fitsizedistcontdata


#############################################
errordbhlikcont <- function(param,fitfcn,minsize,maxsize,sizes)
  # calculate likelihood of size distribution data
  # given what is expected under fitfcn type dbh dist function with parameter values param
  # size distribution data are sent as a vector of individual sizes
  # together with min and max of the truncated distribution 
{
  loglikeach=calclogliksobssizes(param,sizes,minsize,maxsize,fitfcn)
  
  negtotloglik=-sum(loglikeach)
  
  return(negtotloglik)
} # end errordbhlikcont


##################################################
calclogliksobssizes <- function(param, sizes,minsize,maxsize,fitfcn)
  # calculate the log likelihoods of the observed sizes (sizes)
  # under the specified type of size distribution (fitfcn) with specified parameters (param)
  # trunated at minsize, maxsize
{
  # loglikeach are the log likelihoods (probabilities) of each data point (sizes) under the full, untruncated distributions
  # logliketot is the log of the total probability in the truncated distribution
  # logliktrunc are the log likelihoods of each data point under the truncated distributions
  if (fitfcn=="weib") { # param[1] is the shape, param[2] is the scale
    loglikeach <- dweibull(sizes,param[1],param[2],log=T)
    logliketot <- ifelse(is.na(maxsize),log(1 - pweibull(minsize,param[1],param[2])),
                         log(pweibull(maxsize,param[1],param[2])-pweibull(minsize,param[1],param[2])))
  }
  else if (fitfcn=="pow") {  # the "power function" size distribution is a Pareto distribution 
    loglikeach <- log(dpareto(sizes,minsize,param[1]))
    logliketot <- ifelse(is.na(maxsize),log(1-ppareto(minsize,minsize,param[1])),
                         log(ppareto(maxsize,minsize,param[1])-ppareto(minsize,minsize,param[1])))
  }
  else if (fitfcn=="exp") {  # param[1] is the parameter of a negative exponential size distributionb
    loglikeach <- dexp(sizes,param[1],log=T)
    logliketot <- ifelse(is.na(maxsize),log(1 - pexp(minsize,param[1])),
                         log(pexp(maxsize,param[1])-pexp(minsize,param[1])))

  }
  logliktrunc <- loglikeach-logliketot  # account for truncation 
  # (probability under the truncated distribution) = 
  #  (probability under the full distribution) / (total probability in the truncated distribution)
  # log (prob truncated) = log(prob full) - log(total prob)
  
  return(logliktrunc)
} # end calclogliksobssizes


# calculate the Kolmogorov-Smirnov statistic for the difference between 
# the observed cumulative distribution and a particular fitted distribution
calcksstat <- function(param, sizes,minsize,maxsize,fitfcn) {
  allcdf <- data.frame(sizes=c(minsize,sort(sizes)),rank=seq(0,length(sizes)))
  allcdf$obscdf <- allcdf$rank/length(sizes)
  
  if (fitfcn=="weib") { # param[1] is the shape, param[2] is the scale
    cdfmin <- pweibull(minsize,param[1],param[2])
    cdfmax <- ifelse(is.na(maxsize),1,pweibull(maxsize,param[1],param[2]))
    allcdf$fitcdf <- (pweibull(allcdf$sizes,param[1],param[2])-cdfmin)/(cdfmax-cdfmin)
  }
  else if (fitfcn=="pow") {  # the "power function" size distribution is a Pareto distribution 
    cdfmin <- ppareto(minsize,minsize,param[1])
    cdfmax <- ifelse(is.na(maxsize),1,ppareto(maxsize,minsize,param[1]))
    allcdf$fitcdf <- (ppareto(allcdf$sizes,minsize,param[1])-cdfmin)/(cdfmax-cdfmin)
  }
  else if (fitfcn=="exp") {  # param[1] is the parameter of a negative exponential size distributionb
    cdfmin <- pexp(minsize,param[1])
    cdfmax <- ifelse(is.na(maxsize),1,pexp(maxsize,param[1]))
    allcdf$fitcdf <- (pexp(allcdf$sizes,param[1])-cdfmin)/(cdfmax-cdfmin)
  }
  
  ksstat <- max(abs(allcdf$fitcdf-allcdf$obscdf))
  return(ksstat)
} # end calcksstat


usedat <- read.csv("gaps_2014_2019_centroid.csv")
usedat <- usedat[usedat$date !="2016-05-18",]
names(usedat) <- c("id","size","bsunit")

allfullfits <- setfitsizedistcontdata(usedat,minsize=c(2,5,10,25),maxsize=c(487,5e5,NA),fitfcn=c("exp","pow","weib"))
write.table(allfullfits,file="allsizedistfitsnoci.csv",sep=",",row.names=F)

bsfits <- bsfitsizedistcontdata(usedat,minsize=c(2,5,10,25),maxsize=c(5e5),fitfcn=c("exp","pow","weib"),
                                   nbootstraps=1000,alpha1=0.05,alpha2=0.01)
write.table(bsfits,file="sizedistfitswithci.csv",sep=",",row.names=F)

