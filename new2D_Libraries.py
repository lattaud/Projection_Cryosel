#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:28:40 2020

@author: arnaud
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import integrate


def Resolution(e):
    e = np.array(e,dtype=float)
    result = np.ones_like(e) * sigma
    mask = (e>0)
    result[mask] = result[mask] + np.power(e[mask],0.5)
    return result

def smeared1D_from_Er_spectrum(Ephs,dNdEr,
                               sigma_phonon = 100,
                               Q = 1,
                               V = 3,
                               epsilon = 3,
                               scale = 'eV',
                               nsig = 7,
                               npts = 1000):
  
    assert scale in ['eV','eVee']

    if (scale == 'eVee'):
        sigma_phonon = sigma_phonon / (1+V/epsilon)
        
    Ephs = np.array(Ephs,dtype=float)
    
    Er_minintegrale = (Ephs-(nsig*sigma_phonon)) / (1 + Q*V/epsilon)
    if (scale == 'eVee'):
        Er_minintegrale = Er_minintegrale * (1+V/epsilon)
    Er_minintegrale = np.maximum(Er_minintegrale,0)
    #recoil energies are necessarily greater than 0
    
    Er_maxintegrale = (Ephs+(nsig*sigma_phonon)) / (1 + Q*V/epsilon)
    if (scale == 'eVee'):
        Er_maxintegrale = Er_maxintegrale * (1+V/epsilon)

    #no boundary necessary   
    Er_edges = np.linspace(Er_minintegrale,Er_maxintegrale,npts)    
    Er_means = 0.5 * (Er_edges[1:] + Er_edges[:-1])
    dEr = Er_edges[1:] - Er_edges[:-1]    
    


    Ephs_from_Er = Er_means*(1.+Q * V/epsilon)
    if (scale == 'eVee'):
        Ephs_from_Er = Ephs_from_Er / (1+V/epsilon)
        
    gausarg =np.exp(-np.power(Ephs - Ephs_from_Er,2)/(2*np.power(sigma_phonon,2)))
    coef = 1. / (np.sqrt(2*pi)*sigma_phonon)
    result = coef * np.sum(dNdEr(Er_means)*gausarg*dEr,axis=0) 
    return result 



def smeared2D_from_Er_spectrum(Ephs,Eion,dNdEr,
                               sigma_phonon = 100,
                               sigma_ion = 100,
                               Q = 1,
                               V = 3,
                               epsilon = 3,
                               scale = 'eV',
                               nsig = 7,
                               npts = 1000):
    # when Q cste that's easy to get Er from Ephs or Eion but when Q(Er), then formulae are not invertible
    # Riemann integral with npts points. Increasing npts gives more accurate but slower results. npts = 1000 is more than enough
    assert scale in ['eV','eVee']

    if (scale == 'eVee'):
        sigma_phonon = sigma_phonon / (1+V/epsilon)
        
    Ephs = np.array(Ephs,dtype=float)
    Eion = np.array(Eion,dtype=float)
    
    Er_minintegrale_from_heat = (Ephs- (nsig*sigma_phonon)) / (1 + Q*V/epsilon)
    Er_minintegrale_from_ion =  (Eion- (nsig*sigma_ion)   ) / Q
    
    if (scale == 'eVee'):
        Er_minintegrale_from_heat = Er_minintegrale_from_heat * (1+V/epsilon)
    
    Er_minintegrale = np.maximum(Er_minintegrale_from_heat,Er_minintegrale_from_ion)
    Er_minintegrale = np.maximum(Er_minintegrale,0)
    #recoil energies are necessarily greater than 0
    
    Er_maxintegrale_from_heat = (Ephs +(nsig*sigma_phonon)) / (1 + Q*V/epsilon)
    Er_maxintegrale_from_ion  = (Eion +(nsig*sigma_ion)   ) / Q
    Er_maxintegrale = np.minimum(Er_maxintegrale_from_heat,Er_maxintegrale_from_ion)
    if (scale == 'eVee'):
        Er_maxintegrale = Er_maxintegrale * (1+V/epsilon)

    #no boundary necessary   
    Er_edges = np.linspace(Er_minintegrale,Er_maxintegrale,npts)    
    Er_means = 0.5 * (Er_edges[1:] + Er_edges[:-1])
    dEr = Er_edges[1:] - Er_edges[:-1]    

    Ephs_from_Er = Er_means * (1.+Q * V/epsilon)
    if (scale == 'eVee'):
        Ephs_from_Er = Ephs_from_Er / (1+V/epsilon)
        
    Eion_from_Er = Er_means * Q
    
    gausarg_heat =np.exp(-np.power(Ephs - Ephs_from_Er,2)/(2*np.power(sigma_phonon,2)))
    gausarg_ion =np.exp(-np.power(Eion - Eion_from_Er,2)/(2*np.power(sigma_ion,2)))
    
    coef = 1. / (np.sqrt(2*pi)*sigma_phonon*sigma_ion)
    result = coef * np.sum(dNdEr(Er_means)*gausarg_heat*gausarg_ion*dEr,axis=0) 
    return result 

