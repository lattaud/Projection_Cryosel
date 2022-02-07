#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:28:40 2020

@author: arnaud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DP_Parameters:
    def __init__(self, Target="Ge",DPmass=1,Kappa=1e-14):
        self.Target = Target
        #self.Exposure = Exposure # always put 1 kg.day default
        self._Kappa = Kappa
        self._DPfilename = 'DarkPhotons-ERIC/tabulated_hp_rate.txt'
        self._DPdataframe = self.LoadDPfile()
        self._DPmass = DPmass
        self._DPrate = None
        lastchoice = None
        self.SetDPmass(DPmass)

    def UpdateValues(self):
        self.SetDPmass(self._DPmass)
        #calling SetDPmass updates everything since even if the mass was chosen based on massindex it will work

    def SetKappa(self,value):
        previousvalue = self._Kappa
        self._Kappa = value
        self._DPdataframe["rate"] = self._DPdataframe["rate"]*(value**2/previousvalue**2)
        self.UpdateValues()
              
    def GetKappa(self):
        return self._Kappa

    def SetDPmass(self,value): 
        mass,rate = self.LoadDPparameters(mass=value,index=None)
        self._DPmass = mass
        self._DPrate = rate 
          
    def SetDPmassindex(self,value): 
        mass,rate = self.LoadDPparameters(mass=None,index=value)
        self._DPmass = mass
        self._DPrate = rate 
    
    def GetDPmass(self): 
        return self._DPmass
    
    def GetDPmassANDrate(self): 
        return self._DPmass, self._DPrate
    
    def GetDPrate(self): 
        return self._DPrate
    
    def GetDPdataframe(self):
        return self._DPdataframe
        
    def LoadDPfile(self, filename=None):
        if filename==None:
            filename = self._DPfilename
        self._DPdataframe = pd.read_csv(self._DPfilename,names=["mass","rate"],sep=' ',skiprows=2)
        return self._DPdataframe
        
    def PlotRatevsMass(self):
        dt = self._DPdataframe
        plt.loglog(dt['mass'],dt['rate'],label='Dark Photons $\kappa={}$'.format(self._Kappa))
        plt.xlabel('mass [eV]')
        plt.ylabel('Event Rate [/kg/day]')
        plt.legend()
    
    # loads the rate corresponding to the closest mass available
    # if index != None, you can ask for the massindex instead of the mass
    def LoadDPparameters(self,mass,index=None):
        dt = self._DPdataframe
        dtselected = dt.iloc[np.argsort(abs(dt['mass']-mass))[0]] if index==None else dt.iloc[index]
        actualmass,actualrate = dtselected['mass'],dtselected['rate']
        return actualmass, actualrate

    def PlotRatevsMass(self):
        dt = self._DPdataframe
        plt.loglog(dt['mass'],dt['rate'],label='Dark Photons $\kappa={}$'.format(self._Kappa))
        plt.xlabel('Energy [eV]')
        plt.ylabel('Event Rate [/kg/day]')
        plt.legend()
