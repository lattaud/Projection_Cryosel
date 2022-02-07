#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:28:40 2020

@author: arnaud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ALP_Parameters:
    def __init__(self, Target="Ge",Exposure=1,ALPmass=1,gAe=1e-12):
        self.Target = Target
        #self.Exposure = Exposure # always put 1 kg.day default
        self._gAe = gAe
        self._ALPfilename = 'DarkPhotons-ERIC/tabulated_alp_rate.txt'
        self._ALPdataframe = self.LoadALPfile()
        self._ALPmass = ALPmass
        self._ALPrate = None
        lastchoice = None
        self.SetALPmass(ALPmass)

    def UpdateValues(self):
        self.SetALPmass(self._ALPmass)
        #calling SetALPmass updates everything since even if the mass was chosen based on massindex it will work

    def SetgAe(self,value):
        previousvalue = self._gAe
        self._gAe = value
        self._ALPdataframe["rate"] = self._ALPdataframe["rate"]*(value**2/previousvalue**2)
        self.UpdateValues()
              
    def GetgAe(self):
        return self._gAe

    def SetALPmass(self,value): 
        mass,rate = self.LoadALPparameters(mass=value,index=None)
        self._ALPmass = mass
        self._ALPrate = rate 
          
    def SetALPmassindex(self,value): 
        mass,rate = self.LoadALPparameters(mass=None,index=value)
        self._ALPmass = mass
        self._ALPrate = rate 
    
    def GetALPmass(self): 
        return self._ALPmass
    
    def GetALPmassANDrate(self): 
        return self._ALPmass, self._ALPrate
    
    def GetALPrate(self): 
        return self._ALPrate
    
    def GetALPdataframe(self):
        return self._ALPdataframe
        
    def LoadALPfile(self, filename=None):
        if filename==None:
            filename = self._ALPfilename
        self._ALPdataframe = pd.read_csv(self._ALPfilename,names=["mass","rate"],sep=' ',skiprows=2)
        return self._ALPdataframe
        
    def PlotRatevsMass(self):
        dt = self._ALPdataframe
        plt.loglog(dt['mass'],dt['rate'],label='Axions gAe={}'.format(self._gAe))
        plt.xlabel('mass [eV]')
        plt.ylabel('Event Rate [/kg/day]')
        plt.legend()
    
    # loads the rate corresponding to the closest mass available
    # if index != None, you can ask for the massindex instead of the mass
    def LoadALPparameters(self,mass,index=None):
        dt = self._ALPdataframe
        dtselected = dt.iloc[np.argsort(abs(dt['mass']-mass))[0]] if index==None else dt.iloc[index]
        actualmass,actualrate = dtselected['mass'],dtselected['rate']
        return actualmass, actualrate


