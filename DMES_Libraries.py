#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:28:40 2020

@author: arnaud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#foldername = '/sps/edelweis/DMfiles/QEdark-vesc544'

class DMES_Parameters:
    def __init__(self,DMmass=3e7,CrossSection=1e-37):
        self._TableCrossSection = 1e-37
        self._CrossSection = CrossSection
        self._filemasses = 'DMfiles/September-dm-masses-in-eV.txt'
        self._fileprefactor = 'DMfiles/pre-factors-for-dm-masses-rho03.dat'
        self._df_filemasses = self.Load_dataframe_massindex_converter()
        self._DMmass = DMmass
        self._DMtype = 'light' # light or heavy
        self._DMprefactor = None
        self._scalingfactor = 1./ 365.25
        self._DMdataframe = None #self.LoadDMdataframe()
        self._ = None
        lastchoice = None
        self.SetDMmass(DMmass)
        
        
    def SetCrossSection(self,value):
        self._CrossSection = value
        self._scalingfactor = (self._CrossSection**2/self._TableCrossSection**2)/ 365.25  # rates are stored for 1e-37 cm2 and 1 kg.year
        #self.UpdateValues()
        
    def GetDMmass(self): 
        return self._DMmass
    
    def GetCrossSection(self): 
        return self._CrossSection
    
    #def GetDMdataframe(self):
    #    return self._DMdataframe
        
    def SetDMmass(self,value): 
        mass, dataframe, prefactor = self.LoadDMparameters(mass=value,index=None)
        self._DMmass = mass
        self._DMdataframe = dataframe 
        self._DMprefactor = prefactor
          
    def SetDMmassindex(self,value): 
        mass, dataframe, prefactor = self.LoadDMparameters( mass=None,index=value)
        self._DMmass = mass
        self._DMdataframe = dataframe 
        self._DMprefactor = prefactor

    def UpdateValues(self):
        self.SetDMmass(self._DMmass)
        #calling SetDMmass updates everything since even if the mass was chosen based on massindex it will work
        
    def Get_index_from_massindex(self,massindex):
        assert massindex in np.arange(1,53+1,1) # [1,2,3,...,53]
        return massindex-1
    
    def Get_mass_from_massindex(self,massindex): 
        index =  self.Get_index_from_massindex(massindex)
        df_select = self._df_filemasses.iloc[index] 
        assert df_select["massindex"] == massindex
        return df_select["mass"]
    
    def Get_prefactor_from_massindex(self,massindex): 
        index =  self.Get_index_from_massindex(massindex)
        df_select = self._df_filemasses.iloc[index] 
        assert df_select["massindex"] == massindex
        return df_select["prefactor"]
        
    def Get_filename_from_massindex(self,massindex):
        assert massindex in np.arange(1,53+1,1) # [1,2,3,...,53]
        filename = 'DMfiles/QEdark-vesc544/C.{:02d}.dat'.format(massindex) # eg. C.01.dat, C.02.dat, ... , C.53.dat
        return filename
    
    def Load_dataframe_massindex_converter(self):
        df = pd.read_csv(self._filemasses,names=['massindex','mass'],sep='\s+')
        dfprefac = pd.read_csv(self._fileprefactor,names=['prefactor'],sep='\s+')
        dfall = df.join(dfprefac)
        return dfall
   
    def Get_massindex_from_mass(self,mass):
        dt = self._df_filemasses
        dtselected = dt.iloc[np.argsort(abs(dt['mass']-mass))[0]]
        massindex = dtselected['massindex']
        return int(massindex)
    

    def LoadDMparameters(self,mass,index=None):
        massindex = self.Get_massindex_from_mass(mass) if index==None else index
        filename = self.Get_filename_from_massindex(massindex)
        dfall =  pd.read_csv(filename,sep='\s+',skiprows=0,header=None) 
        #self._DMdataframe = dfall
        actualmass = self.Get_mass_from_massindex(massindex)
        prefactor = self.Get_prefactor_from_massindex(massindex)
        return actualmass,dfall,prefactor
            
    def GetDMLinesAndRates(self):
        """
                9 rows of 3+500 values: binned recoil spectra in energy bins (500 0.1eV-bins from 0.0 to 50.0eV)
                The first column stands for the DM-formfaktor (1,2,3: F_DM=1, =1/q, =1/q^2)
                The second column represents the month and so annual modulation of the velocity of the earth (1: December ve_mod=-15kmps, 2: March ve_mod=0, 3: June ve_mod=15kmps)
                The third column is the total rate (arbitrary units!), the following bins represent the binned rate, i.e. 500 0.1eV bins (again, arbitrary units!)
                normalisation factor for rates to correspond to events/kg/year for a cross section of sigma=1e-37 cm2 is 778429.68 (preliminary!!!) 

                For example, for a standard halo velocity and a form factor F_DM=1/q^2, one has to select the combination "3 2" which is the 6th row in the data files.
             0   01 	01
             1   02	01
             2   03	01
             3   01	02	F=1 no modulation
             4   02	02
             5   03	02	F=1/q2 no modulation
             6   01	03
             7   02	03
             8   03	03
        """
        dfall = self._DMdataframe
        loclight = 3
        locheavy = 5
        if self._DMtype == 'light':
            loc = loclight
        elif self._DMtype == 'heavy':
            loc = locheavy
        else:
            print("problem ",self._DMtype," is not a type")
            
        totalrate = dfall.iloc[loc][2]
        totalrate = np.array(totalrate,dtype=float)
        
        rates = dfall.iloc[loc][3:-1]
        rates = np.array(rates,dtype=float)
        sumofrates = np.sum(rates)
        assert np.isclose(totalrate,sumofrates)       
        
        
        rates = rates * self._DMprefactor * self._scalingfactor
    
        # 500 valeurs entre 0 et 500
        e = np.linspace(0,50,501)
        means = 0.5 * (e[1:] + e[0:-1])  
        return means,rates
    
    def PlotSpectrum(self):
        means,rates = self.GetDMLinesAndRates()
        plt.plot(means,rates,label=f"$\sigma=${self._CrossSection} $cm^{2}$")
        plt.xlabel("Energy [eV]")
        plt.ylabel("Events / kg / day / 0.1 eV bin")
        plt.legend(loc='upper right')
