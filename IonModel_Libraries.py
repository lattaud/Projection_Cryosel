#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:32:06 2020

@author: arnaud
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import scipy.stats as ss
#from math import floor,ceil
import math
import random
import time


def GetMeanAndF(xx,proba):
    """
    ---------- Parameters
    xx : np.array of integers         e.g.    xx = np.array(np.arange(0,50,1).astype(int))
    proba : np.array of probabilities e.g. proba = ss.norm.pdf(xx,loc=25,scale=5)
    such that np.sum(proba)==1
    -------Returns
    truemean : mean of distribution 
    trueF : Fano factor of distribution
    """
    tol=1e-4
    #check that the probability distribution is normalized to 1
    assert abs(np.sum(proba) - 1) < tol, "sum of prob="+str(np.sum(proba))+" instead of 1"
    truemean = np.sum(xx * proba)
    truestd = np.sqrt(np.sum(np.power(xx - truemean, 2) * proba)) 
    trueF = truestd**2 / truemean if truemean!=0 else 1

    return truemean,trueF


def get_values_from_Target(Target):
    if Target=="Ge" :
        epsilon = 3
        gap = 0.67
    elif Target=="Si" :
        epsilon = 3.6
        gap = 1.1
    else:
        print("Carefull, This target has not been implemented yet")
        epsilon = 999
        gap = 999
    return epsilon,gap

class Parameters:
    
    def __init__(self, Voltage=100,SigmaeV=10,IonModel="CDMS",Fano=1,Target="Ge"):
        self.Voltage=Voltage
        self.SigmaeV=SigmaeV
        self.Ionmodel=IonModel
        self.Fano= Fano
        self.Target = Target
    
    # VOLTAGE
    @property
    def Voltage(self): 
        return self._Voltage
    
    @Voltage.setter 
    def Voltage(self, value): 
        self._Voltage = value
        
    # sigmaeV (phonon baseline resolution RMS in eV)
    @property
    def SigmaeV(self): 
        return self._SigmaeV
    
    @SigmaeV.setter 
    def SigmaeV(self, value): 
        self._SigmaeV = value
    
    # sigmaeVee (phonon baseline resolution RMS in eVee)  no setter
    @property
    def SigmaeVee(self): 
        return self._SigmaeV/(1.+self._Voltage/3.)
    
    
    
    # epsilon and gap are from target. no setter
    @property
    def Epsilon(self): 
        epsilon,gap = get_values_from_Target(self.Target)
        return epsilon
    
    @property
    def Gap(self): 
        epsilon,gap = get_values_from_Target(self.Target)
        return gap
    
    
    @property
    def Fano(self): 
        return self._Fano
    
    @Fano.setter 
    def Fano(self, value): 
        self._Fano = value
 
    
    @property
    def IonModel(self): 
        return self._IonModel
 
    @IonModel.setter 
    def IonModel(self, value): 
        self._IonModel = value
        
    def Information(self):
        print("\n")
        print("Voltage = ",self.Voltage, "V")
        print("sigmaeV = ",self.SigmaeV, "eV")
        print("sigmaeVee = ",self.SigmaeVee, "eVee")
        print("Fano = ",self.Fano)
        print("Target = ",self.Target)
        print("epsilon = ",self.Epsilon," eV")
        print("gap = ",self.Gap," eV")
        
   
         
def Prob_Ion_single(N,E,DET): 

    N = np.asarray(N)
    # unpack usefull values from DET
    ionmodel = DET.IonModel
    F = DET.Fano
    epsilon = DET.Epsilon 
    gap = DET.Gap
    assert (ionmodel=="Rouven") | (ionmodel=="CDMS")
    
    if ionmodel=="CDMS":
        
        if(E<gap):
            return 1 if N==0 else 0
        elif(gap <= E < epsilon):
            return 1 if N==1 else 0
        else:
            mu = E/epsilon
            return Prob_DWB(N,mu,F)
                
    elif ionmodel=="Rouven":
        
        if(E<gap):
            return 1 if N==0 else 0
        else:
            floorvalue=1+np.floor((E-gap)/epsilon)
            return 1 if (N==floorvalue) else 0
        
        
def Prob_Ion_arrayN(N,E,DET): 

    N = np.array(N,dtype=int)
    # unpack usefull values from DET
    ionmodel = DET.IonModel
    F = DET.Fano
    epsilon = DET.Epsilon 
    gap = DET.Gap
    assert (ionmodel=="Rouven") | (ionmodel=="CDMS")
    
    prob = np.zeros_like(N,dtype=float)
    
    if ionmodel=="CDMS":
        if(E<gap):
            prob = np.where(N==0,1,0)
            
        elif(gap <= E < epsilon):
            prob = np.where(N==1,1,0)
        
        else:
            mean = E/epsilon
            prob = Prob_DWB(N,mean,F)#ss.poisson.pmf(N,mean)    # on prend poisson pour simplifier
    
    elif ionmodel=="Rouven":
        if(E<gap):
             prob = np.where(N==0,1,0)
        else:
            floorvalue=1+np.floor((E-gap)/epsilon)
            prob = np.where(N==floorvalue,1,0)
    else:
        print(ionmodel,"is not a viable option")
    return prob

def Prob_Ion_arrayE(N,E,DET): 

    N = np.asarray(N,dtype=int)
    E = np.asarray(E,dtype=float)

    # unpack usefull values from DET
    ionmodel = DET.IonModel
    F = DET.Fano
    epsilon = DET.Epsilon 
    gap = DET.Gap
    assert (ionmodel=="Rouven") | (ionmodel=="CDMS")
    
    prob = np.zeros_like(E).astype(dtype=float)
    if ionmodel=="CDMS":
        
        maskcond1 = (E<gap)
        prob[maskcond1] = 1 if N==0 else 0
        
        maskcond2 = (gap <= E) & (E < epsilon) 
        prob[maskcond2] = 1 if N==1 else 0 
        
        maskcond3 = ~(maskcond1 | maskcond2) 
        mean = E[maskcond3] / epsilon # CAREFULL, need to apply maskcond3 to E
        prob[maskcond3] = Prob_DWB_arraymu(N,mean,F)#ss.poisson.pmf(N,mean)  #on prend poisson pour simplifier
    
    elif ionmodel=="Rouven":
        maskcond1 = (E<gap)
        prob[maskcond1] = 1 if N==0 else 0
        
        maskcond2 = ~ maskcond1
        floorvalue= 1 + np.floor((E[maskcond2]-gap)/epsilon)
        prob[maskcond2] = np.where(N==floorvalue,1,0)
     
    else:
        print(ionmodel,"is not a viable option")
        
    return prob

def Prob_Ion_array_multi(N,E,DET): 

    N = np.asarray(N,dtype=int)
    E = np.asarray(E,dtype=float)
    assert (N.size == E.size)
    
    # unpack usefull values from DET
    ionmodel = DET.IonModel
    F = DET.Fano
    epsilon = DET.Epsilon 
    gap = DET.Gap
    assert (ionmodel=="Rouven") | (ionmodel=="CDMS")
    
    prob = np.zeros_like(N).astype(dtype=float)
    
    if ionmodel=="CDMS":
        
        maskcond1 = (E<gap)
        prob[maskcond1] = np.where(N[maskcond1]==0,1,0)
        
        maskcond2 = (gap <= E) & (E < epsilon) #forced to separate conditions
        prob[maskcond2] = np.where(N[maskcond2]==1,1,0)
        
        maskcond3 = ~(maskcond1 | maskcond2) # should be equivalent to a else
        mean = E[maskcond3] / epsilon # CAREFULL, need to apply maskcond3 to E
        prob[maskcond3] = Prob_DWB_arraymu(N[maskcond3],mean,F)#ss.poisson.pmf(N[maskcond3],mean) # CAREFULL, need to apply maskcond3 to N, already added to 
    
    elif ionmodel=="Rouven":
        maskcond1 = (E<gap)
        prob[maskcond1] =  np.where(N[maskcond1]==0,1,0)
        
        maskcond2 = ~ maskcond1
        floorvalue= 1 + np.floor((E[maskcond2]-gap)/epsilon)
        prob[maskcond2] = np.where(N[maskcond2]==floorvalue,1,0)
     
    else:
        print(ionmodel,"is not a viable option")
        
    return prob

def Prob_Ion(N,E,DET): 

    
    N=np.asarray(N,dtype=int)
    E=np.asarray(E,dtype=float)
    Nsize = N.size
    Esize = E.size
    if (Nsize==1) & (Esize==1):
        return Prob_Ion_single(N,E,DET)
    elif (Nsize>1) & (Esize==1):
        return Prob_Ion_arrayN(N,E,DET)
    elif (Nsize==1) & (Esize>1):
        return Prob_Ion_arrayE(N,E,DET)
    elif Nsize==Esize:
        return Prob_Ion_array_multi(N,E,DET)
    else:
        print("this is not possible")
        assert (Nsize!=1) & (Esize!=1) & (Nsize!=Esize)
    
    

def Bernoulli_F(mu):
    Bernoulli_Nmax=np.ceil(mu)
    Bernoulli_Nmin=np.floor(mu)
    p = Bernoulli_Nmax - mu
    with np.errstate(divide='ignore',invalid='ignore'):
        mode = p*(1.-p)/mu
    return mode

def Bernoulli_Prob(k,mu):
    k = np.asarray(k,dtype=int)
    mu = np.asarray(mu,dtype=float)

    size_k, size_mu = k.size, mu.size
    assert (size_k==1) | (size_mu==1) | (size_mu==size_k)
    
    Bernoulli_Nmax=np.ceil(mu)
    Bernoulli_Nmin=np.floor(mu)
    p = Bernoulli_Nmax - mu
    
    size_prob = max(size_k,size_mu) # 
    prob = np.zeros_like(size_prob,dtype=float) # size of mu
    prob = prob + (k==Bernoulli_Nmin) * p
    prob = prob + (k==Bernoulli_Nmax) * (1-p)
    return prob

def Prob_DWB(k,mu,F,switch=200):    
    """
    WORKS for any size of k but mu must be of size 1
    """
    k = np.asarray(k)
    assert (F<=1) & (F>=0) & (mu>=0)
    
    if F == 1:
        return ss.poisson.pmf(k,mu)
    
    if mu == 0:
        return (k==0) * 1
        
    
    Fmin = Bernoulli_F(mu)
    # less or equal to within tolerance is important because of computational rounding errors
    if (F<=Fmin) | (abs(F-Fmin)<1e-5):
        return Bernoulli_Prob(k,mu)
    
    # switch to Gaussian when mu > switch
    if mu > switch:   
        return ss.norm.pdf(k,loc=mu,scale=np.sqrt(F*mu))
	
    nl = np.floor(mu/(1.-F)).astype(int)
    nh = np.ceil(mu/(1.-F)).astype(int)
    assert (nl!=0) & (nh!=0)      
    Fl = 1-mu/nl
    Fh = 1-mu/nh        
    
    if (nl==nh):
        return ss.binom.pmf(k=k,n=nl,p=1-Fl)    
    
    assert F>Fmin        
    if (Fl<=0) & (Fh>0):

        Bernoulli_Nmax=np.ceil(mu)
        Bernoulli_Nmin=np.floor(mu)
        p=Bernoulli_Nmax-mu
        Fb=p*(1.-p)/mu        
        #selection_Weighted = (F<Fmin)
        
        #if F<Fmin: # weighted 
        DeltaF = (F-Fb)/(Fh-Fb)
        wb = (1.-DeltaF)
        wh = DeltaF
        Binomial_h = ss.binom.pmf(k=k,n=nh,p=1-Fh)
        
        if(np.floor(mu) == np.ceil(mu)):
            Proba_Bernoulli = (k==mu) * 1  #special case 
        else:
            Proba_Bernoulli = Bernoulli_Prob(k,mu)
            
        Weighted= wb*Proba_Bernoulli + wh*Binomial_h
        return Weighted
    
    else:
        DeltaF=(F-Fl)/(Fh-Fl)
        wl=(1.-DeltaF)
        wh=DeltaF
        Binomial_l=ss.binom.pmf(k=k,n=nl,p=1-Fl) 
        Binomial_h=ss.binom.pmf(k=k,n=nh,p=1-Fh) 
        Weighted= wl*Binomial_l + wh*Binomial_h
        return Weighted
    
def Prob_DWB_arraymu(k,mu,F,switch=200):   
    with np.errstate(divide='ignore',invalid='ignore'):
        """
        trying to make it work for any siez of mu, with k of size 1
        """
        k = np.asarray(k,dtype=int)
        mu = np.asarray(mu,dtype=float)

        size_k, size_mu = k.size, mu.size
        assert (size_k==1) | (size_mu==1) | (size_mu==size_k), " sk="+str(size_k)+" smu="+str(size_mu)
        assert (F<=1) & (F>=0) #& np.all(mu>=0)

        if F == 1:
            return ss.poisson.pmf(k,mu)

        size_prob = max(size_k,size_mu)
        #
        prob = np.zeros(size_prob,dtype=float)
        resultfound = np.zeros(size_prob,dtype=bool) # put a 1 instead of 0 for element where prob is already calculated


        mask = np.logical_and(mu<0,resultfound==False)
        prob = np.where(mask,0,prob)
        resultfound[mask] = True

        # case mu = 0  return 1 if N=0 else 0
        mask = np.logical_and(mu==0,resultfound==False)
        prob = np.where(mask,np.where((k==0),1,0),prob) #changes values of prob only for True elements in mask1
        resultfound[mask] = True
        if resultfound.all():
            return prob

        Fmin = Bernoulli_F(mu)
        # less or equal to within tolerance is important because of computational rounding errors
        mask = np.logical_and((F<=Fmin) | (abs(F-Fmin)<1e-5),resultfound==False)
        prob = np.where(mask,Bernoulli_Prob(k,mu),prob)
        resultfound[mask] = True
        if resultfound.all():
            return prob

        # switch to Gaussian when mu > switch
        mask = np.logical_and(mu>switch,resultfound==False)
        prob = np.where(mask,ss.norm.pdf(k,loc=mu,scale=np.sqrt(F*mu)),prob)
        resultfound[mask] = True
        if resultfound.all():
            return prob


        mask = (resultfound==False)
        nl = np.where(mask,np.floor(mu/(1.-F)).astype(int),0)
        nh = np.where(mask,np.ceil(mu/(1.-F)).astype(int),0)

        maskassert = np.logical_or( (nl!=0) & (nh!=0),resultfound==True)
        #assert maskassert.all()
        #assert (nl!=0) & (nh!=0),"nl="+str(nl)+"  nh="+str(nh)+"  k="+str(k)+"  mu="+str(mu)+"  F="+str(F)+" Fmin="+str(Fmin)        
        # idealy i would like to calculate Fl and Fh only for elements that have not been found yet
        # but since np.where performs the operation everywhere, alocate arbitrary value 999 to elements that will not be used
        Fl = np.where(resultfound==False,1-mu/nl,999)
        Fh = np.where(resultfound==False,1-mu/nh,999)

        mask = np.logical_and(nl==nh,resultfound==False)
        prob = np.where(mask,ss.binom.pmf(k=k,n=nl,p=1-Fl),prob)
        resultfound[mask] = True
        if resultfound.all():
            return prob

        maskcond1 = np.logical_and((Fl<=0) & (Fh>0),resultfound==False)

        Bernoulli_Nmax=np.ceil(mu)
        Bernoulli_Nmin=np.floor(mu)
        p=Bernoulli_Nmax-mu
        Fb=p*(1.-p)/mu       
        DeltaF = (F-Fb)/(Fh-Fb)
        wb = (1.-DeltaF)
        wh = DeltaF
        Binomial_h = ss.binom.pmf(k=k,n=nh,p=1-Fh) # might be missing k[mask5]   for multi 


        Y_maskcondfloor = (np.floor(mu) == np.ceil(mu))
        N_maskcondfloor = ~Y_maskcondfloor
        maskcond11 = maskcond1 & Y_maskcondfloor
        maskcond12 = maskcond1 & N_maskcondfloor

        Proba_Bernoulli = np.zeros(size_prob,dtype=float)
        Proba_Bernoulli = np.where(maskcond11,(k==mu)*1,Proba_Bernoulli)
        Proba_Bernoulli = np.where(maskcond12, Bernoulli_Prob(k,mu),Proba_Bernoulli)

        prob=np.where(maskcond1,wb*Proba_Bernoulli + wh*Binomial_h,prob)
        resultfound[maskcond1] = True
        if resultfound.all():
            return prob


        # all that remains
        mask = (resultfound==False)
        DeltaF=(F-Fl)/(Fh-Fl)
        wl=(1.-DeltaF)
        wh=DeltaF
        Binomial_l=ss.binom.pmf(k=k,n=nl,p=1-Fl) 
        Binomial_h=ss.binom.pmf(k=k,n=nh,p=1-Fh) 
        prob = np.where(mask,wl*Binomial_l + wh*Binomial_h,prob)
    return prob

    
def compare_speed_function(f,x,y):
    # assumes the function is of the form f(x,y)
    print("\n")
    print("Compare speed function")
    x = np.asarray(x)
    y = np.asarray(y)
    size = len(x)
    assert size==len(y)
    res1 = []
    res2 = []
    res3 = []
    # evaluation element by element
    print("For the evaluation of NxN elements : [f(x1,y1),f(x2,y1),f(xN,y1),....f(x1,y2),f(x1,yN)]")
    print(str(size)+"x"+str(size)+" elements")
    start = time.time()
    for j in range(size):
        for i in range(size):
            result = f(x[i],y[j])
            res1.append(result)
    end=time.time()
    totaltime1 = end-start  
    print("{:2.4f}".format(totaltime1)+" s  to evaluate the "+str(size)+"x"+str(size)+" elements one by one")

    res1 = np.array(res1)
    #print("res1",res1)
    start = time.time()
    for j in range(size):
       result = f(x,y[j])
       res2.append(result)
    end=time.time()
    totaltime2 = end-start  
    print("{:2.4f}".format(totaltime2)+" s  ({:2.2f} faster) with x as array : [f(x,y1),f(x,y2),...,f(x,yN)]".format(totaltime1/totaltime2))
    res2 = np.array(res2)
    res2 = res2.ravel()
    #print("res2",res2)
    print("results all same = ",np.allclose(res1,res2))
    
    start = time.time()
    for j in range(size):
       result = f(x,y[j])
       res3.append(result)
    end=time.time()
    totaltime3 = end-start  
    print("{:2.4f}".format(totaltime3)+" s  ({:2.2f} faster) with y as array : [f(x1,y),f(x2,y),...,f(xN,y)]".format(totaltime1/totaltime3))
    res3 = np.array(res3)
    res3 = res3.ravel()
    #print("res3",res3)
    print("results all same = ",np.allclose(res1,res3))
    
    res1bis = []
    start = time.time()
    for i in range(size):
        result = f(x[i],y[i])
        res1bis.append(result)
    end=time.time()
    totaltime1bis = end-start  
    print("")
    print("For the evaluation of N elements : [f(x1,y1),f(x2,y2),....f(xN,yN)]")
    print("{:2.4f}".format(totaltime1bis)+" s  to evaluate the "+str(size)+" elements one by one")
    res1bis = np.array(res1bis)
    
    start = time.time()
    res4 = f(x,y)
    end = time.time()
    totaltime4 = end-start  
    print("{:2.4f}".format(totaltime4)+" s  ({:2.2f} faster) with x and y as arrays and bitwise operation ".format(totaltime1bis/totaltime4))
    res4 = np.array(res4)
    res4 = res4.ravel()
    print("results all same = ",np.allclose(res1bis,res4))

def compare_speed_function_withargs(f,x,y,*args):
    Nargs = len(args)
    if(Nargs==0):
        compare_speed_function(f,x,y)
    elif(Nargs==1):
        ff = lambda xx,yy: f(xx,yy,args[0]) 
        compare_speed_function(ff,x,y)
    else:
        print("not implemented")
        
        
        
