#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:31:50 2020

@author: arnaud
"""
import scipy.stats as ss
from scipy import special, interpolate, integrate
import numpy as np
import os.path
from os import path
from scipy.stats import chi2
import matplotlib.pyplot as plt
import time
#from mytools import *
#from toolbox import * 
from Tools_Libraries import *
    
class LimitSetter:
    def __init__(self,CL=90,Nvalues=1000,rewrite=False):
        self._CL = CL
        self._Nvalues = Nvalues
        self.LoadPoissonLimitDataFile(ConfidenceLevel=CL,Nvalues = Nvalues,rewrite=rewrite)
        
        
    @property
    def CL(self): 
        return self._CL
    
    @CL.setter
    def CL(self,value): 
        self._CL = value
        self.LoadPoissonLimitDataFile(ConfidenceLevel=self._CL,Nvalues = self._Nvalues, rewrite=True)
    
    @property
    def Nvalues(self): 
        return self._Nvalues
    
    @Nvalues.setter
    def Nvalues(self,value): 
        self._Nvalues = value
        self.LoadPoissonLimitDataFile(ConfidenceLevel=self._CL,Nvalues = self._Nvalues, rewrite=True)
    
    
    @property
    def Precalculatedvalues(self): 
        return self._Precalculatedvalues
    
    # @Precalculatedvalues.setter
    # def Target(self, value):
    #     self._Precalculatedvalues = value    
        
    def LoadPoissonLimitDataFile(self,ConfidenceLevel=None,Nvalues = None, rewrite=None):
        filename="PoissonDataFile-CL{0}-{1}values.txt".format(ConfidenceLevel,Nvalues) 
        if (path.exists(filename)==True) & (rewrite==False):
            print("Existing PoissonDataFile with {:d} precalculated values is used".format(Nvalues))
        else:
            print("Creating PoissonDataFiles")
            with open(filename, 'w') as f:
                for i in range(int(Nvalues)):
                    muexcl_fromPoisson = self.muexcluded(i,ConfidenceLevel,Gausoption=False)
                    # muexcl_fromGaus = muexcluded(i,ConfidenceLevel,Gausoption=True)
                    #print(i, muexcl_fromPoisson, muexcl_fromGaus)
                    print("{} {}".format(i,muexcl_fromPoisson))
                    f.write("{} {} \n".format(i,muexcl_fromPoisson))
            
        Counter = 0
        result = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                N, Nexcl = line.rstrip().split()
                assert int(N) == int(Counter), "{} {}".format(N,Counter)
                result.append(Nexcl)
                Counter+=1
        result = np.array(result,dtype=float)
        self._Precalculatedvalues = result
        #return result
        
    def PoissonLimitMoyenne(self,vecmuobs = None):
        def singlepass(muobs):
            frozenPoissonFunction = ss.poisson(muobs)
            nmin,nmax = frozenPoissonFunction.interval(0.9999)
            #print(nmin,nmax)
            Ns = np.array(list(range(int(nmin),int(nmax+1))))
            return np.sum(frozenPoissonFunction.pmf(Ns) * self.PoissonLimit(Ns))
        
        if(type(vecmuobs)==np.ndarray):
            size = vecmuobs.size
            return np.array([singlepass(muobs) for muobs in vecmuobs])
        else:
            return singlepass(vecmuobs)
        
    def PoissonLimitMediane(self,vecmuobs = None):
        return self.PoissonLimit(ss.poisson.median(vecmuobs))
        
    def PoissonLimitFast(self,Nobs = None):
        #Approximation only. e.g.   Nobs = 10.2 will return 0.8 * PoissonLimit(10) + 0.2 * PoissonLimit(11)
        # only to be used for debugging. Otherwise, use PoissonLimitMoyenne
        f = np.floor(Nobs)
        c = np.ceil(Nobs)
        diff = Nobs - f
        ones = np.ones_like(Nobs,dtype=float)
        result = (ones - diff) * self.PoissonLimit(f) + diff * self.PoissonLimit(c)
        return result
    
    
    def PoissonLimit(self,Nobs = None):
        switch = self.Nvalues - 1 # calculated only from 0 to N - 1
        Nobs = np.asarray(Nobs,dtype=int)
        if Nobs.size==1:
            if(Nobs<switch):
                return self.muexcl_from_Poisson(Nobs)
            else:
                return self.muexcl_from_Chi2(Nobs)
        else:
            results = np.zeros_like(Nobs,dtype=float)
            
            maskcond1 = (Nobs<switch)
            IDscond1 = np.where(maskcond1)
            onlycond1 = Nobs[IDscond1]
            resultsonlycond1 = self.muexcl_from_Poisson(onlycond1)
            results[IDscond1] = resultsonlycond1
            
            maskcond2 = ~maskcond1
            IDscond2 = np.where(maskcond2)
            onlycond2 = Nobs[IDscond2]
            resultsonlycond2 = self.muexcl_from_Chi2(onlycond2)
            results[IDscond2] = resultsonlycond2
            
            return results
        
    def CL_from_Poisson(self,mu,Nobs):
        integers_from_0_to_Nobs = np.array(range(0,int(Nobs+1)))
        #print(integers_from_0_to_Nobs)
        #print(ss.poisson.pmf(integers_from_0_to_Nobs,mu))
        result= np.sum(ss.poisson.pmf(integers_from_0_to_Nobs,mu))
        return 100.*(1.-result)


    def CL_from_Gaus(self,mu,Nobs):
        alpha = 0.5 * special.erfc( (mu-Nobs) / np.sqrt(2*mu))
        return 100.*(1.-alpha)
    
    def CL_from_Chi2(self,mu,Nobs):
        alpha = chi2.cdf(2*mu,df=(2*Nobs+1))
        return 100.*(alpha)
    
    
    def muexcludedold(self,Nobs,ConfidenceLevel = None,Gausoption=False):
        Npts = 1e4
        logoption = False if Nobs<10 else True
    
        if logoption == True:     
            start = np.log10(Nobs)
            xs = np.logspace(start,start+2,int(Npts))
        else:
            xs = np.linspace(0,100,int(Npts))
        
        if (Gausoption == True) :
            vecCL = [self.CL_from_Gaus(x,Nobs) if x>10 else self.CL_from_Poisson(x,Nobs) for x in xs ]
        else:
            vecCL = [self.CL_from_Poisson(x,Nobs) for x in xs]
        vecCL = np.asarray(vecCL,dtype=float)
        g = interpolate.interp1d(vecCL,xs)
        return g(ConfidenceLevel)
    
    def muexcluded(self,Nobs,ConfidenceLevel = None,Gausoption=False,tol=0.01):
        xmin = max(Nobs,1)
        xmax = max(Nobs * 10,10)
        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        diff = 100
        count = 0
        while (abs(diff)>tol):
            count+=1
            deltalog = (logmax-logmin)/2
            log = logmin+deltalog
            x = np.power(10,logmin+deltalog)
            result = self.CL_from_Poisson(x,Nobs) 
            if (Gausoption == True) :
                result = self.CL_from_Gaus(x,Nobs)
            else:
                result = self.CL_from_Poisson(x,Nobs) 
            diff = result-ConfidenceLevel
            if(diff>0):
                logmax = logmin+deltalog
            else:
                logmin = logmin+deltalog
        return x#,count

            
    def muexcl_from_Chi2(self,Nobs):
        Nobs = np.asarray(Nobs,dtype=int)
        alpha = 1 - self._CL/100.
        return chi2.interval(1-2*alpha,df=2*Nobs+np.ones_like(Nobs))[1]/2. # we get 2muexcl from df = 2 N + 1
    
    def muexcl_from_Poisson(self,Nobs = None):
        assert np.logical_or(type(Nobs) == int,isinstance(Nobs,np.ndarray)) # Nobs is necessarily an Integer for a single experiment
        return self._Precalculatedvalues[Nobs]
    
    def DetermineROI(self,fs=None,fb=None,ultimateminROI=None,ultimatemaxROI=None,nptx=None,Plot=False):
        if nptx == None:
            nptx = 100

        xedges = np.linspace(ultimateminROI,ultimatemaxROI,nptx)
            
        f_s = fs(xedges)
        f_b = fb(xedges)
        
        cumul_s = integrate.cumtrapz(f_s,xedges,initial=0)
        cumul_b = integrate.cumtrapz(f_b,xedges,initial=0)
                
        integral_s = cumul_s[-1]
        integral_b = cumul_b[-1]
        
        cumul_eff_s = cumul_s/integral_s
        cumul_eff_b = cumul_b/integral_b
        
        sens_init = self.PoissonLimitMediane(integral_b) # signal and background efficiencies == 1 
        #print(sens_init)
        bestsensimprovement = 1
        minROIfound = ultimateminROI
        maxROIfound = ultimatemaxROI
        sizeROIfound = maxROIfound - minROIfound

        minROIplot = []
        maxROIplot = []
        improvementplot = []
        
        for minID,minROI in enumerate(xedges):
            for maxID,maxROI in enumerate(xedges):
                if(maxID<=minID):
                    continue
                else:
                    sizeROI = maxROI - minROI
                    mu_b = cumul_b[maxID] - cumul_b[minID]
                    eff_s = cumul_eff_s[maxID] - cumul_eff_s[minID]
                    limit_mu_b = self.PoissonLimitMediane(mu_b)
                    assert (eff_s<=1)
                    sensimprovement = sens_init/limit_mu_b*eff_s
                    
                    minROIplot.append(minROI)
                    maxROIplot.append(maxROI)
                    improvementplot.append(sensimprovement)
                    #print(mu_b,limit_mu_b,eff_s,sensimprovement)
                    
                    condition1 = (sensimprovement>bestsensimprovement)
                    condition2 = np.logical_and(sensimprovement==bestsensimprovement,sizeROI>sizeROIfound)
                    if np.logical_or(condition1,condition2):
                        bestsensimprovement = sensimprovement
                        minROIfound = minROI
                        maxROIfound = maxROI
                        sizeROIfound = maxROIfound - minROIfound
                        
        #print(bestsensimprovement,"for min = ",minROIfound," and ",maxROIfound)
        if(Plot):
            f, (ax1,ax2) = plt.subplots(2,1, figsize=(5.5,9),constrained_layout=True)
            ax1.plot(xedges,f_s,color="C0",label="Signal")
            ax1.set_xlabel("Energy [eV]")
            ax1.plot(xedges,f_b,color="C1",label="Background")
            ax1.vlines(minROIfound, ymin=-1, ymax=30, color='r')
            ax1.vlines(maxROIfound, ymin=-1, ymax=30, color='r')
            ax1.legend()  
            #print("starting plot")
            monscatterplot = ax2.scatter(minROIplot,maxROIplot,c=improvementplot,cmap="RdBu_r")
            ax2.set_xlabel("Emin ROI")
            ax2.set_ylabel("Emax ROI")
            cbar = plt.colorbar(monscatterplot)
            cbar.set_label("Improvement factor")
            #print("done")
        return minROIfound,maxROIfound
    
    def NEW_DetermineROI(self,fs=None,fb=None,ultimateminROI=None,ultimatemaxROI=None,nptx=None,Plot=False):
        if nptx == None:
            nptx = 100
        
        xedges = np.linspace(ultimateminROI,ultimatemaxROI,nptx)
            
        f_s = fs(xedges)
        f_b = fb(xedges)
        
        cumul_s = integrate.cumtrapz(f_s,xedges,initial=0)
        cumul_b = integrate.cumtrapz(f_b,xedges,initial=0)
        
        integral_s = cumul_s[-1]
        integral_b = cumul_b[-1]
        
        cumul_eff_s = cumul_s/integral_s
        cumul_eff_b = cumul_b/integral_b
        
        sens_init = self.PoissonLimitMediane(integral_b) # signal and background efficiencies == 1 
       
        IDS = [ID for ID, val in enumerate(xedges)]
        IDx,IDy = np.meshgrid(IDS,IDS)
        
        IDx = IDx.flatten() # omg i jus had to do that....
        IDy = IDy.flatten() # omg i jus had to do that....
        
        def maxsensandmaxROI(vecsens,ROImin,ROImax,vecsize):
            mask = (vecsens == vecsens.max())
            equalIDs = np.where(mask)[0]            
            shortvecsens = vecsens[equalIDs]
            shortvecsize = vecsize[equalIDs]
            shortvecsizeDO = np.argmax(shortvecsize)
           
            winnerID = equalIDs[shortvecsizeDO]
            winnersens = vecsens[winnerID]
            winnersize = vecsize[winnerID]
            winnerROImin = ROImin[winnerID]
            winnerROImax = ROImax[winnerID]

            return winnersens,winnerROImin,winnerROImax,winnersize
        
        def mafonction(minID,maxID):
            #result = np.zeros_like(minID,dtype=float)
            vecemin = xedges[minID]
            vecemax = xedges[maxID]
            vecsizeROI = vecemax-vecemin
            masknull = (minID > maxID)
            mu_b = np.where(~masknull,cumul_b[maxID] - cumul_b[minID],0)
            eff_s = cumul_eff_s[maxID] - cumul_eff_s[minID]
            #mask_emin_lowerthan_emax = (IDmin<IDmax)
            limit_mu_b = self.PoissonLimitMediane(mu_b)
            #assert (eff_s<=1)
            
            vecsensimprovement = sens_init/limit_mu_b*eff_s
            vecminROIfound = vecemin
            vecmaxROIfound = vecemax
            
            
            vecminROIfound[masknull] = 0
            vecmaxROIfound[masknull] = 0
            vecsensimprovement[masknull] = 0
            
            vecsizeROIfound = vecmaxROIfound - vecminROIfound
            
            
            
            return vecsensimprovement,vecminROIfound,vecmaxROIfound,vecsizeROIfound


        vecsens,vecminROI,vecmaxROI,vecsizeROI = mafonction(IDx,IDy)
        bestsensimprovement, minROIfound, maxROIfound,sizeROIfound = maxsensandmaxROI(vecsens,vecminROI,vecmaxROI,vecsizeROI)
        sizeROIfound = maxROIfound - minROIfound
        #bestID = np.unravel_index(np.argmax(vecsens, axis=None), vecsens.shape)
        #minROIfound = vecminROI[bestID]
        #maxROIfound = vecmaxROI[bestID]
        #sizeROIfound = vecsizeROI[bestID]
        #bestsensimprovement = vecsens[bestID]
        
        #print(bestsensimprovement,"for min = ",minROIfound," and ",maxROIfound)
        if(Plot):
            f, (ax1,ax2) = plt.subplots(2,1,figsize=(5.5,9), constrained_layout=True)
            ax1.plot(xedges,f_s,color="C0",label="Signal")
            ax1.plot(xedges,f_b,color="C1",label="Background")
            ax1.vlines(minROIfound, ymin=-1, ymax=30, color='r')
            ax1.vlines(maxROIfound, ymin=-1, ymax=30, color='r')
            ax1.set_xlabel("Energy [eV]")
            ax1.legend()
            monscatterplot = ax2.scatter(xedges[IDx],xedges[IDy],c=vecsens,cmap="RdBu_r")
            #monscatterplot = ax2.scatter(minROIfound,maxROIfound,c=bestsensimprovement,cmap="RdBu_r")
            ax2.set_xlabel("Emin ROI")
            ax2.set_ylabel("Emax ROI")
            cbar = plt.colorbar(monscatterplot)
            cbar.set_label("Improvement factor")
            #f, (ax1,ax2) = plt.subplots(2,1, figsize=(5.5,9),constrained_layout=True)
            #ax1.plot(xedges,f_s,color="C0",label="Signal")
            #ax1.set_xlabel("Energy [eV]")
            #ax1.plot(xedges,f_b,color="C1",label="Background")
            #ax1.vlines(minROIfound, ymin=-1, ymax=30, color='r')
            #ax1.vlines(maxROIfound, ymin=-1, ymax=30, color='r')
            #ax1.legend()  
            #monscatterplot = ax2.scatter(minROIplot,maxROIplot,c=improvementplot,cmap="RdBu_r")
            #ax2.set_xlabel("Emin ROI")
            #ax2.set_ylabel("Emax ROI")
            #cbar = plt.colorbar(monscatterplot)
            #cbar.set_label("Improvement factor")
        return minROIfound,maxROIfound
    
    def Multi_DetermineROI(self,fs=None,fb=None,ultimateminROI=None,ultimatemaxROI=None,nptx=None,Plot=False):
        
        xedges = np.linspace(ultimateminROI,ultimatemaxROI,nptx)
        xmeans = 0.5*(xedges[:-1]+xedges[1:])
        dx = xedges[1:]-xedges[:-1]
    
        x = xmeans

        Ns = fs(x)*dx
        Nb = fb(x)*dx
        
        totals = np.sum(Ns)
        totalb = np.sum(Nb)

        effs = Ns/totals
        #print(effs)
        #assert (effs>0)
        effb = Nb/totalb
        
        scoretobeat = self.PoissonLimitMediane(totalb)
        
        soverb = effs/effb
        IDs = np.argsort(soverb)[::-1]
        
        Ns_DO = Ns[IDs] # Decreasing order of SoverB (similar to increasing order of BoverS)
        Nb_DO = Nb[IDs]
        soverb_DO = soverb[IDs]
        effs_DO = effs[IDs]
        
        Limit = self.PoissonLimitMediane(np.cumsum(Nb_DO))
        Limiteffcorrected = Limit/np.cumsum(effs_DO)
        LimitImprovement = scoretobeat/Limiteffcorrected
        
        
        # we want to select the biggest ROI in case of similar sensitivity improvement
        # so we can't just  maxID = np.argmax(LimitImprovement)  which is a scalar
        mask = (LimitImprovement == LimitImprovement.max())
        equalIDs = np.where(mask)[0]  
        maxID = equalIDs[-1]
        # Done
        
        IDsSelected = IDs[0:maxID+1]
        
        xselected = xmeans[IDsSelected]
        dxselected = dx[IDsSelected]
        CouplesEminEmax = list(zip(xselected-dxselected/2,xselected+dxselected/2))
        CouplesSorted = sorted(CouplesEminEmax, key = lambda x: x[0]) # tri des couples (Ein,Emax) par ordre croissant de Emin
        
        malist = np.array(CouplesSorted).flatten()
        redundantIDs = [(i,i+1) for i in range(len(malist)-1) if np.isclose(malist[i],malist[i+1])]
        x = np.delete(malist,redundantIDs)
        ROIlist = [(x[i],x[i+1]) for i in range(0,len(x)-1,2)]
        #print("ROIlist=",ROIlist)
        
        # ROImin, ROImax = CouplesSorted[0]
        # _ , ULTRA_ROImax = CouplesSorted[-1]
        # ROIlist = []
        
        # for (atomin,atomax) in CouplesSorted[1:]:
        #     condition1 = np.isclose(atomin,ROImax)
        #     condition2 = np.isclose(atomax,ULTRA_ROImax)
        #     if np.logical_or(condition1,condition2):
        #         ROImax = atomax
        #         if(condition2):
        #             ROIlist.append((ROImin,ROImax))
        #     else:
        #         ROIlist.append((ROImin,ROImax))
        #         ROImin = atomin
        #         ROImax = atomax  
        print("final ROI list consists of",len(ROIlist),"ROI(s)")
        for toto,tata in ROIlist:
            print("[{:.2f},{:.2f}] eV".format(toto,tata))  
        
        if Plot:
            #plt.plot(x,fs(x),"ro")
            Finalmub = 0
            Finaleffs = 0
            for eminfound,emaxfound in ROIlist:
                Finalmub += TF1Integral(fb,eminfound,emaxfound)
                Finaleffs += TF1Integral(fs,eminfound,emaxfound)/totals 
                
            FinalImprovement = scoretobeat/self.PoissonLimitMediane(Finalmub)*Finaleffs
            print(FinalImprovement,"for the following ROI(s) found")
            #plt.plot(LimitImprovement)
            #mask = (LimitImprovement == LimitImprovement.max())
            #equalIDs = np.where(mask)[0]  
            #print(equalIDs)  
        return ROIlist



        

