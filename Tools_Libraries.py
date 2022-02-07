#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:22:45 2020

@author: arnaud
"""

import math
import numpy as np
import scipy.stats as ss
from scipy.integrate import quad,quad_vec
from scipy.interpolate import interp1d
from scipy import signal
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

def TF1Integral(TF1=None,Emin=None,Emax=None,Nptx=1000):  
    logoption = False
    if Emin>0:
        if (Emax/Emin)>Nptx:
            logoption = True
    if logoption==True:
        x = np.logspace(np.log10(Emin),np.log10(Emax),Nptx) 
    else:
        x = np.linspace(Emin,Emax,Nptx) 
    return np.trapz(TF1(x),x)

def TF1IntegralMulti(TF1=None,ROIlist=None,Nptx=1000):  
    """
    ROIlist should be like : [(Emin1,Emax1),(Emin2,Emax2),....]
    TF1IntegralMultiROI will return the 
    """
    return sum(TF1Integral(TF1,Emin,Emax,Nptx) for Emin,Emax in ROIlist)

    
def Gaus_smearing(Es,f1,sigma,nsig=10,method="ultime_quad2",precision=100,onlypositive=False): 
    Es = np.asarray(Es,dtype=float)
    result = []

    dx = sigma/precision
    xmin = Es[0] - nsig *sigma
    # if onlypositive:
    #     xmin = max(xmin,0.) #pour gerer le fait que le spectre en energie est défini que pour les énergies positives
    #print("a enlever")
    xmax = Es[len(Es)-1] + nsig * sigma
    mean = (xmin+xmax)/2.
    N = np.ceil((xmax - xmin)/dx).astype(int)
    xx = np.linspace(xmin,xmax,N)
    dx = (xx[1]-xx[0])
    #print("xmin=",xmin,"xmax=",xmax)
    if(method=="ultimate"):
        # requires Energies to be given in increasing order
        f2 = np.exp(-(xx-mean)**2/(2*sigma**2))/(np.sqrt(2*math.pi)*sigma)
        f3 = f1(xx)#,args)
        conv = np.convolve(f3,f2,mode='same')  * dx
        inter = interp1d(xx, conv,kind='linear')#,kind="linear")
        return np.array([inter(a) for a in Es])
    elif(method=="ultimate_ss"):
        # requires Energies to be given in increasing order
        f2 = ss.norm.pdf(xx,loc=mean,scale=sigma)
        f3 = f1(xx)#,args)
        conv = np.convolve(f3,f2,mode='same') * dx
        inter = interp1d(xx, conv)
        return np.array([inter(a) for a in Es])
    elif(method=="gaussian_filter1d"):
        # requires Energies to be given in increasing order
        f3 = f1(xx) #,args)
        # if onlypositive==True:
        #     f3 = lambda z : f3(z) * (z>0) + 1e-19
        conv = gaussian_filter1d(f3,sigma/dx,truncate=nsig)
        #print(gaussian_filter1d)
        inter = interp1d(xx, conv)
        return np.array([inter(a) for a in Es])
    elif(method=="ultime_quad"):
        # requires Energies to be given in increasing order
        func_to_integrate = lambda x : f1(x)*ss.norm.pdf(x,loc=Es,scale=sigma)
        res = quad_vec(func_to_integrate,xmin,xmax,quadrature="gk21")[0]
        return res
    elif(method=="ultime_quad2"):
        # requires Energies to be given in increasing order
        func_to_integrate = lambda x : f1(x)*ss.norm.pdf(x,loc=xx,scale=sigma)
        res = quad_vec(func_to_integrate,xmin,xmax)[0]
        inter = interp1d(xx, res)
        return np.array([inter(a) for a in Es])
    else:
        for i in range(Es.size):
            E = Es[i]
            xmin = E - nsig * sigma
            xmax = E + nsig * sigma
            if(method=="quad_ss"):
                func_to_integrate = lambda x : f1(x)*ss.norm.pdf(x,loc=E,scale=sigma)
                tot = quad(func_to_integrate,xmin,xmax)[0]
                
            elif(method=="quad_0ss"):
                func_to_integrate = lambda x : f1(x)*np.exp(-(E-x)**2/(2*sigma**2))    
                tot = quad(func_to_integrate,xmin,xmax)[0] * 1./(np.sqrt(2*math.pi)*sigma) 
                
            elif(method=="quad_0ss_0pow"):
                func_to_integrate = lambda x : f1(x)*np.exp(-(E-x)*(E-x)/(2*sigma*sigma))    
                tot = quad(func_to_integrate,xmin,xmax)[0] * 1./(np.sqrt(2*math.pi)*sigma) 
                
            elif(method=="Own_Rieman_ss"):
                #dx = sigma/precision
                N = np.ceil((xmax - xmin)/dx).astype(int)
                x_midpoint = np.linspace(xmin+dx/2,xmax-dx/2,N)
                midpoint_riemann_sum = np.sum(f1(x_midpoint) * dx *ss.norm.pdf(E-x_midpoint,scale=sigma))
                tot = midpoint_riemann_sum
                
            elif(method=="Own_Rieman_0ss"):
                #dx = sigma/precision
                N = np.ceil((xmax - xmin)/dx).astype(int)
                x_midpoint = np.linspace(xmin+dx/2,xmax-dx/2,N)
                midpoint_riemann_sum = np.sum(f1(x_midpoint) * dx *np.exp(-(E-x_midpoint)**2/(2*sigma**2)))
                tot = midpoint_riemann_sum * 1./(np.sqrt(2*math.pi)*sigma)  
            elif(method=="Rieman_Trapz"):
                #dx = sigma/precision
                N = np.ceil((xmax - xmin)/dx).astype(int)
                x_midpoint = np.linspace(xmin+dx/2,xmax-dx/2,N)
                midpoint_riemann_sum = f1(x_midpoint) *np.exp(-(E-x_midpoint)**2/(2*sigma**2)) * 1./(np.sqrt(2*math.pi)*sigma)
                tot = np.trapz(midpoint_riemann_sum,x_midpoint) 
            else:
                print('this method is not implemented yet')  
                
            result.append(tot)
        result = np.asarray(result,dtype=float)
    return result
"""
def smear(xs,functosmear,kernel='gaus',scale=1): 
    xs = np.asarray(xs,dtype=float)
    if(kernel=='gaus'):
        funckernel = lambda x : ss.norm.pdf(x,loc=xs,scale=sigma)
    else:
        print("problem")
    integrand = lambda x : functosmear(x)*f1(x)*ss.norm.pdf(x,loc=Es,scale=sigma)  
    res = quad_vec(integrand,xmin,xmax)[0]
    return res
"""